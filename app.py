from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from flask_cors import cross_origin
import io
from PIL import Image
import torch
from watermark_anything.data.metrics import msg_predict_inference
import os
from torchvision import transforms
from datetime import datetime

from notebooks.inference_utils import (
    load_model_from_checkpoint,
    create_random_mask,
    unnormalize_img,
    plot_outputs,
    msg2str
)

# 정규화 파라미터 (ImageNet 기준)
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

# 원본 크기 유지 + 정규화만 적용하는 transform
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std),
])

app = Flask(__name__)
CORS(app)

# 모델 준비 (서버 시작 시 1회만)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "checkpoints/wam_mit.pth"
json_path = "checkpoints/params.json"
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

# 결과 폴더 경로 지정
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return "서버 구동 완료~"

@app.route('/watermark-insert', methods=['POST'])
def watermarkInsert():
    # 이미지와 메시지 받기
    image_file = request.files['image']
    message = request.form.get('message', 'AI24')
    assert len(message) <= 4, "메시지는 4자 이하만 가능"

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

    # 1. 실제 삽입 마스크 (GT) 생성 (원본 이미지와 동일 크기)
    mask_gt = mask.squeeze().cpu().numpy()  # (1, H, W) → (H, W)
    mask_gt_pil = Image.fromarray((mask_gt * 255).astype('uint8'))

    # 2. 마스크 이미지 저장
    base_name = os.path.splitext(image_file.filename)[0]
    mask_gt_filename = f"{base_name}_mask_gt.png"
    mask_gt_path = os.path.join(RESULTS_DIR, mask_gt_filename)
    mask_gt_pil.save(mask_gt_path)

    # 메시지 저장
    message_save_path = os.path.join(RESULTS_DIR, f"{base_name}_message.txt")
    with open(message_save_path, 'w') as f:
        f.write(message)


    # 1. 서버에 파일로 저장
    original_name = os.path.splitext(image_file.filename)[0]  # 확장자 제외
    filename = f"{original_name}_deeptruth_watermark.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    out_img_pil.save(save_path)

    # 클라이언트 응답에 GT 마스크 URL 추가
    return jsonify({
        "message": "워터마크 삽입 성공",
        "download_url": f"/results/{filename}",
        "mask_gt_url": f"/results/{mask_gt_filename}",
    })

@app.route('/watermark-detection', methods=['POST'])
def watermarkDetection():
    try:
        # 1. 이미지 수신 및 기본 정보 추출
        image_file = request.files['image']
        original_filename = image_file.filename
        base_name = original_filename.split('_deeptruth_watermark')[0]
        
        # 2. GT 마스크 & 원본 메시지 파일 경로 생성
        mask_gt_filename = f"{base_name}_mask_gt.png"
        mask_gt_path = os.path.join(RESULTS_DIR, mask_gt_filename)
        message_filename = f"{base_name}_message.txt"
        message_path = os.path.join(RESULTS_DIR, message_filename)

        # 3. 필수 파일 존재 여부 확인
        if not all(os.path.exists(p) for p in [mask_gt_path, message_path]):
            return jsonify({"error": "워터마크 정보를 찾을 수 없습니다."}), 404

        # 4. 원본 메시지 로드
        with open(message_path, 'r') as f:
            original_message = f.read().strip()

        # 5. 이미지 & 마스크 전처리
        image = Image.open(image_file.stream).convert("RGB")
        img_pt = default_transform(image).unsqueeze(0).to(device)
        
        # 마스크를 업로드 이미지 크기로 맞추기
        mask_gt = Image.open(mask_gt_path).convert('L')
        mask_gt = mask_gt.resize(image.size, resample=Image.NEAREST)
        mask_gt = transforms.ToTensor()(mask_gt).unsqueeze(0).to(device)
        mask_gt = (mask_gt > 0.5).float()  # 이진화

        # 6. 메시지 추출
        with torch.no_grad():
            detect_outputs = wam.detect(img_pt)  # <-- .extract() 대신 .detect()
            preds = detect_outputs['preds']      # shape: [B, 1+nbits, H, W]
            mask_preds = preds[:, 0:1, :, :]     # 첫 채널은 마스크
            bit_preds = preds[:, 1:, :, :]       # 나머지는 비트 메시지

        # 7. 정확도 계산
        pred_message = msg_predict_inference(bit_preds, mask_preds)
        restored_msg = msg2str(pred_message[0])
        
        # 8. 원본 메시지 텐서 변환
        wm_bits = ''.join(f"{ord(c):08b}" for c in original_message.ljust(4, '\x00'))[:32]
        wm_tensor = torch.tensor([int(b) for b in wm_bits], dtype=torch.float32).to(device)
        
        bit_acc = (pred_message == wm_tensor.unsqueeze(0)).float().mean().item()

        # 9. 업로드 이미지 저장
        upload_filename = f"{base_name}_detection_input.png"
        upload_path = os.path.join(RESULTS_DIR, upload_filename)
        image.save(upload_path)

        # 10. 날짜 생성
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            "uploaded_image": f"/results/{upload_filename}",
            "mask_gt": f"/results/{mask_gt_filename}",
            "original_message": original_message,
            "bit_accuracy": float(f"{bit_acc * 100:.1f}"),
            "detected_at": timestamp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results/<filename>', methods=['GET'])
@cross_origin(origins="*")
def get_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)