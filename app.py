from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import torch
from watermark_anything.data.metrics import msg_predict_inference
import os, re
from torchvision import transforms
from datetime import datetime
import base64
import requests, time

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

# 이미지 전송 진행률 함수

# BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080") 

SPRING_SERVER_URL = 'http://localhost:8080/progress' 

def send_progress_to_spring(task_id, percent):
    try:
        payload = {
            'taskId': task_id,
            'progress': percent
        }
        headers = {
            'Content-Type': 'application/json'
        }
        print(f"Flask에서 Spring으로 POST 요청 보내는 중: {payload}", flush=True)
        requests.post(SPRING_SERVER_URL, json=payload, headers=headers, timeout=1)
    except Exception as e:
        print(f"[WARN] 진행률 전송 실패: {e}")

# def send_progress_to_spring(task_id, progress):
#     print("send_progress_to_spring 함수 진입")
#     url = 'http://localhost:8080/progress'
#     data = {
#         "taskId": task_id,
#         "progress": progress
#     }
#     response = requests.post(url, json=data)
#     return response.json()


# 파일명(한글 포함)
def safe_filename(filename: str) -> str:
    # 확장자 분리
    name, ext = os.path.splitext(filename)
    # 한글, 영문, 숫자, 일부 특수문자만 허용 → 나머지는 제거
    name = re.sub(r'[^가-힣a-zA-Z0-9_\- ]', '', name)
    # 공백을 _ 로 변환
    name = name.replace(" ", "_")
    return name + ext

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
    task_id = request.form.get('taskId') # taskId 받아오기

    # 이미지와 메시지 받기
    image_file = request.files.get('image')
    message = request.form.get('message', 'AI24')
    assert len(message) <= 4, "메시지는 4자 이하만 가능"
    if not image_file or not message:
        return jsonify({"error": "image, message 둘 다 필요합니다."}), 400
    
    # 이미지 전송 시작
    print("워터마크 삽입 함수 시작")
    send_progress_to_spring(task_id, 0)
    print("send_progress_to_spring 호출 후")

    # 이미지 로드 및 전처리
    image = Image.open(image_file.stream).convert("RGB")
    img_pt = default_transform(image).unsqueeze(0).to(device)

    # 메시지 변환
    wm_bits = ''.join(f"{ord(c):08b}" for c in message)
    wm_bits = wm_bits.ljust(32, '0')[:32]
    wm_msg = torch.tensor([[int(bit) for bit in wm_bits]], dtype=torch.float32).to(device)

    # 워터마크 삽입
    outputs = wam.embed(img_pt, wm_msg)
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.5)
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
    send_progress_to_spring(task_id, 50)

    # 1. 정규화 해제 + 값 범위 제한 (0~1)
    out_img = unnormalize_img(img_w).squeeze(0).detach().clamp_(0, 1)

    # 2. CPU로 이동 후 numpy 변환 (HWC 형태)
    out_img_np = out_img.permute(1, 2, 0).cpu().numpy()

    # 3. 0~255 범위로 변환 (소수점 처리 개선)
    out_img_np = (out_img_np * 255).round().astype('uint8')

    # 4. PIL 이미지 생성
    out_img_pil = Image.fromarray(out_img_np)

    # # 1. 삽입 마스크 (GT) 생성 (원본 이미지와 동일 크기)
    # mask_gt = mask.squeeze().cpu().numpy()  # (1, H, W) → (H, W)
    # mask_gt_pil = Image.fromarray((mask_gt * 255).astype('uint8'))

    # 파일명 처리
    original_name = os.path.splitext(safe_filename(image_file.filename))[0]  # example.jpg → example
    ext = os.path.splitext(safe_filename(image_file.filename))[1]            # 확장자 (jpg, png 등)
    watermarked_name = f"{original_name}_deeptruth_watermark{ext}"           # 파일명 (확장자 포함)

    # 이미지 전송 완료
    send_progress_to_spring(task_id, 100)

    response = jsonify({
        'image_base64': pil_to_base64(out_img_pil),     # 삽입 이미지
        'message': message,                             # 워터마크 메세지
        'filename': watermarked_name,                   # 다운로드 시 사용 될 파일 이름
        # 'mask_image_base64': pil_to_base64(mask_gt_pil) # 마스크 이미지,
        'taskId': task_id
    })
    return response

# 워터마크 탐지
@app.route('/watermark-detection', methods=['POST'])
def watermarkDetection():
    try:
        task_id = request.form.get('taskId') # taskId 받아오기
        # 1. 이미지 수신 및 기본 정보 추출
        image_file = request.files.get('image')
        message = request.form.get('message', '')               # 삽입 당시 메시지 (db에서 가져오는 값)
        # mask_gt_base64 = request.form.get('mask_gt_base64')     # 삽입 당시 마스크 이미지(base64, db에서 가져오는 값)

        send_progress_to_spring(task_id, 0)

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

        send_progress_to_spring(task_id, 50)
        
        # 8. 원본 메시지 텐서 변환
        wm_bits = ''.join(f"{ord(c):08b}" for c in message.ljust(4, '\x00'))[:32]
        wm_tensor = torch.tensor([int(b) for b in wm_bits], dtype=torch.float32).to(device)
        
        bit_acc = (pred_message == wm_tensor.unsqueeze(0)).float().mean().item()
        bit_acc_pct = round(bit_acc * 100, 1)

        # 10. 응답
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        original_name = os.path.splitext(safe_filename(image_file.filename))[0]  # example.jpg → example
        ext = os.path.splitext(safe_filename(image_file.filename))[1]  
        base_name = f"{original_name}{ext}"

        send_progress_to_spring(task_id, 100)

        # 기본 결과값 (정확도 90이상 시)
        result = {
            "basename": base_name,
            "bit_accuracy": bit_acc_pct,
            "detected_at": timestamp,
            'taskId': task_id
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