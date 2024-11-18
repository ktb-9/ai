import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import requests
from io import BytesIO
import time
import signal

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)

# 모델 로드 (글로벌 변수로 설정)
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 타임아웃 예외 및 핸들러
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)

@app.route('/edit-image', methods=['POST'])
def edit_image():
    # 요청 타임아웃 설정 (60초)
    signal.alarm(60)
    try:
        data = request.json
        image_url = data.get('image_url')
        instruction = data.get('instruction')

        # instruction 값 확인
        if not instruction:
            return jsonify({"error": "Instruction cannot be empty. Provide a valid value."}), 400

        print(f"Received image_url: {image_url}")
        print(f"Received instruction: {instruction}")

        # 이미지 다운로드
        start_time = time.time()
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image. HTTP Status Code: {response.status_code}")
            return jsonify({"error": "Failed to download image"}), 400
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image downloaded successfully in {time.time() - start_time:.2f} seconds.")

        # 이미지 크기 제한
        original_size = image.size
        image = image.resize((512, 512), Image.LANCZOS)
        print(f"Image resized from {original_size} to {image.size}.")

        # 모델 처리 시간 측정
        start_time = time.time()
        edited_images = pipe(prompt=instruction, image=image).images
        print(f"Model processing completed in {time.time() - start_time:.2f} seconds.")

        # 수정된 이미지 저장
        output_path = "edited_image.jpg"
        edited_images[0].save(output_path)
        print(f"Edited image saved as {output_path}.")

        # 타임아웃 해제
        signal.alarm(0)
        return jsonify({"message": "Image edited successfully", "output_image": output_path}), 200
    except TimeoutException:
        print("Request timed out.")
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)