from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import torch
from watermark_anything.data.metrics import msg_predict_inference
import os
from torchvision import transforms
from datetime import datetime
import base64

from notebooks.inference_utils import (
    load_model_from_checkpoint,
    create_random_mask,
    unnormalize_img,
)

# 정규화 파라미터 (ImageNet 기준)
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

# 원본 크기 유지 + 정규화만 적용하는 transform
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std),
])

# base64 변환 함수
def pil_to_base64(pil_img, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

app = Flask(__name__)
CORS(app, origins="*")

# 모델 준비 (서버 시작 시 1회만)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "checkpoints/wam_mit.pth"
json_path = "checkpoints/params.json"
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

@app.route('/', methods=['GET'])
def home():
    return "서버 구동 완료~"

# 워터마크 삽입
@app.route('/watermark-insert', methods=['POST'])
def watermarkInsert():
    # 이미지와 메시지 받기
    # image_file = request.files['image']
    image_file = request.files.get('image')
    message = request.form.get('message', 'AI24')
    assert len(message) <= 4, "메시지는 4자 이하만 가능"
    if not image_file or not message:
        return jsonify({"error": "image, message 둘 다 필요합니다."}), 400

    # 이미지 로드 및 전처리
    image = Image.open(image_file.stream).convert("RGB")
    img_pt = default_transform(image).unsqueeze(0).to(device)

    # 메시지 변환
    wm_bits = ''.join(f"{ord(c):08b}" for c in message)
    wm_bits = wm_bits.ljust(32, '0')[:32]
    wm_msg = torch.tensor([[int(bit) for bit in wm_bits]], dtype=torch.float32).to(device)

    # 워터마크 삽입
    outputs = wam.embed(img_pt, wm_msg)
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.1)
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)

    # 1. 정규화 해제 + 값 범위 제한 (0~1)
    out_img = unnormalize_img(img_w).squeeze(0).detach().clamp_(0, 1)

    # 2. CPU로 이동 후 numpy 변환 (HWC 형태)
    out_img_np = out_img.permute(1, 2, 0).cpu().numpy()

    # 3. 0~255 범위로 변환 (소수점 처리 개선)
    out_img_np = (out_img_np * 255).round().astype('uint8')

    # 4. PIL 이미지 생성
    out_img_pil = Image.fromarray(out_img_np)

    # 1. 삽입 마스크 (GT) 생성 (원본 이미지와 동일 크기)
    # mask_gt = mask.squeeze().cpu().numpy()  # (1, H, W) → (H, W)
    # mask_gt_pil = Image.fromarray((mask_gt * 255).astype('uint8'))

    response = jsonify({
        'image_base64': pil_to_base64(out_img_pil),     # 삽입 이미지
        'message': message
    })
    return response

# 워터마크 탐지
@app.route('/watermark-detection', methods=['POST'])
def watermarkDetection():
    try:
        # 1. 이미지 수신 및 기본 정보 추출
        # image_file = request.files['image']
        image_file = request.files.get('image')
        message = request.form.get('message', '')               # 삽입 당시 메시지 (db에서 가져오는 값)
        # mask_gt_base64 = request.form.get('mask_gt_base64')     # 삽입 당시 마스크 이미지(base64, db에서 가져오는 값)

        if not image_file or not message:
            return jsonify({"error": "image, message 둘 다 필요합니다."}), 400

        # 5. 이미지 & 마스크 전처리
        image = Image.open(image_file.stream).convert("RGB")
        img_pt = default_transform(image).unsqueeze(0).to(device)

        # GT 마스크 복원 -> 업로드 이미지 크기로 리사이즈
        # mask_gt_pil = base64_to_pil(mask_gt_base64, mode="L").resize(image.size, resample=Image.NEAREST)
        # mask_gt = transforms.ToTensor()(mask_gt_pil).unsqueeze(0).to(device)
        # mask_gt = (mask_gt > 0.5).float()

        # 6. 추론
        with torch.no_grad():
            detect_outputs = wam.detect(img_pt)  # <-- .extract() 대신 .detect()
            preds = detect_outputs['preds']      # shape: [B, 1+nbits, H, W]
            mask_preds = preds[:, 0:1, :, :]     # 첫 채널은 마스크
            bit_preds = preds[:, 1:, :, :]       # 나머지는 비트 메시지

        # 7. 정확도 계산
        pred_message = msg_predict_inference(bit_preds, mask_preds)
        
        # 8. 원본 메시지 텐서 변환
        wm_bits = ''.join(f"{ord(c):08b}" for c in message.ljust(4, '\x00'))[:32]
        wm_tensor = torch.tensor([int(b) for b in wm_bits], dtype=torch.float32).to(device)
        
        bit_acc = (pred_message == wm_tensor.unsqueeze(0)).float().mean().item()
        bit_acc_pct = round(bit_acc * 100, 1)

        # 10. 응답
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        base_name = os.path.splitext(image_file.filename)[0]

        # 기본 결과값 (정확도 90이상 시)
        result = {
            "basename": base_name,
            "bit_accuracy": bit_acc_pct,
            "detected_at": timestamp,
        }

        # 정확도 < 90이면 삽입 이미지 포함
        if result['bit_accuracy'] < 90:
            result['image_base64'] = pil_to_base64(image)        # 삽입 이미지
            # result['mask_base64'] = pil_to_base64(mask_gt_pil)   # 마스크 이미지

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)