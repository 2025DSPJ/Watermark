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
import requests
import numpy as np, random

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

# SEED 고정
SEED = 42

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# base64 변환 함수
def pil_to_base64(pil_img, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# 이미지 전송 진행률 함수
spring_ip = os.getenv('SPRING_SERVER_IP')
SPRING_SERVER_URL = f'http://{spring_ip}:8080/progress' 

def send_progress_to_spring(task_id, percent, login_id):
    try:
        payload = {
            'taskId': task_id,
            'progress': percent,
            'loginId': login_id
        }
        headers = {
            'Content-Type': 'application/json'
        }
        print(f"Flask에서 Spring으로 POST 요청 보내는 중: {payload}", flush=True)
        requests.post(SPRING_SERVER_URL, json=payload, headers=headers, timeout=1)
    except Exception as e:
        print(f"[WARN] 진행률 전송 실패: {e}")

class ProgressSender:
    def __init__(self, task_id, login_id):
        self.task_id = task_id
        self.login_id = login_id

    def send(self, percent):
        send_progress_to_spring(self.task_id, percent, self.login_id)

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

# =========================================================
# 💡 디버그 코드 추가 지점: 가중치 로드 성공 여부 확인
# =========================================================
try:
    # 모델의 첫 번째 컨볼루션 레이어의 가중치 값을 출력합니다. 
    # 모델 구조에 맞게 'encoder.conv1.weight'를 수정해야 할 수 있습니다.
    first_layer_weights = wam.state_dict()['encoder.conv1.weight']
    
    print(f"[MODEL DEBUG] WAM Layer Shape: {first_layer_weights.shape}")
    print(f"[MODEL DEBUG] WAM First 5 Weights: {first_layer_weights.flatten()[:5].tolist()}", flush=True)
    
except KeyError:
    print("[MODEL DEBUG] WARNING: Cannot find 'encoder.conv1.weight'. Check layer name.", flush=True)
except Exception as e:
    # 이 로그가 찍힌다면, 가중치 로드 자체가 실패했을 가능성이 매우 높습니다.
    print(f"[MODEL DEBUG] CRITICAL ERROR: Failed to read WAM state_dict: {e}", flush=True)

num_threads = torch.get_num_threads()
print(f"현재 PyTorch 기본 스레드 수: {num_threads}")

cpu_count = os.cpu_count()
print(f"CPU 코어 수: {cpu_count}")

@app.route('/', methods=['GET'])
def home():
    return "서버 구동 완료~"

# 워터마크 삽입
@app.route('/watermark-insert', methods=['POST'])
def watermarkInsert():
    task_id = request.form.get('taskId') # taskId 받아오기
    login_id = request.form.get('loginId') # loginId 받아오기

    send_progress = ProgressSender(task_id, login_id)

    # 1. 이미지와 메시지 받기
    image_file = request.files.get('image')
    message = request.form.get('message', 'ETNL')
    assert len(message) <= 4, "메시지는 4자 이하만 가능"
    if not image_file or not message:
        return jsonify({"error": "image, message 둘 다 필요합니다."}), 400
    
    # 작업 진행 상태 초기화
    send_progress.send(0)

    # 2. 이미지 로드 및 전처리
    image = Image.open(image_file.stream).convert("RGB")
    img_pt = default_transform(image).unsqueeze(0).to(device)

    # 3. 메시지 전처리
    wm_bits = ''.join(f"{ord(c):08b}" for c in message)
    wm_bits = wm_bits.ljust(32, '0')[:32]
    wm_msg = torch.tensor([[int(bit) for bit in wm_bits]], dtype=torch.float32).to(device)
    
    # 진행 상태 25%로 업데이트
    send_progress.send(25)

    # 3. 워터마크 삽입
    outputs = wam.embed(img_pt, wm_msg)
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.5)
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)

    # 진행 상태 50%로 업데이트
    send_progress.send(50)

    # 4. 이미지 후처리 
    out_img = unnormalize_img(img_w).squeeze(0).detach().clamp_(0, 1)  # 1. 정규화 해제 + 값 범위 제한 (0~1)
    out_img_np = out_img.permute(1, 2, 0).cpu().numpy()                # 2. CPU로 이동 후 numpy 변환 (HWC 형태)
    out_img_np = (out_img_np * 255).round().astype('uint8')            # 3. 0~255 범위로 변환 (소수점 처리 개선)
    out_img_pil = Image.fromarray(out_img_np)                          # 4. PIL 이미지 생성
    
    # 진행 상태 75%로 업데이트
    send_progress.send(75)

    # 파일명 처리
    original_name = os.path.splitext(safe_filename(image_file.filename))[0]  # example.jpg → example
    ext = os.path.splitext(safe_filename(image_file.filename))[1]            # 확장자 (jpg, png 등)
    watermarked_name = f"{original_name}_deeptruth_watermark{ext}"           # 파일명 (확장자 포함)
    
    # 이미지 전송 완료
    send_progress.send(100)

    response = jsonify({
        'image_base64': pil_to_base64(out_img_pil),     # 삽입 이미지
        'message': message,                             # 워터마크 메세지
        'filename': watermarked_name,                   # 다운로드 시 사용 될 파일 이름
        'taskId': task_id
    })
    return response

# 워터마크 탐지
@app.route('/watermark-detection', methods=['POST'])
def watermarkDetection():
    try:
        task_id = request.form.get('taskId') # taskId 받아오기
        login_id = request.form.get('loginId') # loginId 받아오기

        send_progress = ProgressSender(task_id, login_id)

        # 1. 이미지 수신 및 기본 정보 추출
        image_file = request.files.get('image')
        message = request.form.get('message', '')               # 삽입 당시 메시지 (db에서 가져오는 값)
        if not image_file or not message:
            return jsonify({"error": "image, message 둘 다 필요합니다."}), 400
        
        # 작업 진행 상태 초기화
        send_progress.send(0)

        # 2. 이미지 전처리
        image = Image.open(image_file.stream).convert("RGB")
        img_pt = default_transform(image).unsqueeze(0).to(device)

        # 진행 상태 25%로 업데이트
        send_progress.send(25)

        # 3. 워터마크 탐지 (모델 추론)
        with torch.no_grad():
            detect_outputs = wam.detect(img_pt)
            preds = detect_outputs['preds']      # shape: [B, 1+nbits, H, W]
            mask_preds = preds[:, 0:1, :, :]     # 예측된 마스크
            bit_preds = preds[:, 1:, :, :]       # 예측된 메시지 비트

        # 4. 예측된 비트로부터 메시지 추출
        pred_message = msg_predict_inference(bit_preds, mask_preds)
        pred_message_float = pred_message.float()  # float32로 변환

        # 📌 [ACCURACY DEBUG] 예측 메시지 로그
        print(f"[ACCURACY DEBUG] 4. 예측 메시지 (pred_message) shape: {pred_message.shape}, device: {pred_message.device}")
        print(f"[ACCURACY DEBUG] 예측 비트(첫 8개): {pred_message[0, :8].tolist()}")
        
        # 진행 상태 50%로 업데이트
        send_progress.send(50)
        
        # 5. 원본 메시지 텐서 변환
        wm_bits = ''.join(f"{ord(c):08b}" for c in message.ljust(4, '\x00'))[:32]
        wm_tensor = torch.tensor([int(b) for b in wm_bits], dtype=torch.float32).to(device)

        # 📌 [ACCURACY DEBUG] 원본 메시지 로그
        print(f"[ACCURACY DEBUG] 5. 원본 메시지: '{message}' -> 비트 문자열 길이: {len(wm_bits)}")
        print(f"[ACCURACY DEBUG] 원본 비트(wm_tensor) shape: {wm_tensor.shape}, device: {wm_tensor.device}")
        print(f"[ACCURACY DEBUG] 원본 비트(첫 8개): {wm_tensor[:8].tolist()}", flush=True)

        # comparison_tensor = (pred_message == wm_tensor.unsqueeze(0)).float()
        comparison_tensor = (pred_message_float == wm_tensor.unsqueeze(0)).float()

        # 📌 [ACCURACY DEBUG] 비교 로그
        num_correct_bits = comparison_tensor.sum().item()
        print(f"[ACCURACY DEBUG] 일치하는 비트 수: {num_correct_bits} / 32", flush=True)

        # 6. 비트 정확도 계산
        # bit_acc = (pred_message == wm_tensor.unsqueeze(0)).float().mean().item()
        bit_acc = (pred_message_float == wm_tensor.unsqueeze(0)).float().mean().item()
        bit_acc_pct = round(bit_acc * 100, 1)

        # 📌 [ACCURACY DEBUG] 최종 정확도 로그
        print(f"[ACCURACY DEBUG] 최종 비트 정확도 (bit_acc): {bit_acc_pct}%", flush=True)

        # 진행 상태 75%로 업데이트
        send_progress.send(75)

        # 10. 응답
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        original_name = os.path.splitext(safe_filename(image_file.filename))[0]  # example.jpg → example
        ext = os.path.splitext(safe_filename(image_file.filename))[1]  
        base_name = f"{original_name}{ext}"

        # 이미지 전송 완료
        send_progress.send(100)

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

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)