import streamlit as st
import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
from dotenv import load_dotenv
from utils.lama_cleaner_helper import load_jit_model
import tritonclient.http
from utils.device_utils import get_device

# 전역 디바이스 설정 (GPU 사용 가능시 GPU, 아니면 CPU)
device = get_device()

def get_triton_client():
    """
    Triton 추론 서버 클라이언트를 생성하는 함수
    
    Returns:
        tritonclient.http.InferenceServerClient: Triton 클라이언트 인스턴스
    """
    load_dotenv()  # 환경변수 로드
    url = os.getenv("TRITON_HTTP_URL")  # Triton 서버 URL 가져오기
    
    # Triton 클라이언트 생성
    triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)  
      
    return triton_client

@st.cache_resource
def get_sd_inpaint():
    """
    Stable Diffusion Inpainting 모델을 로드하고 캐싱하는 함수
    
    Returns:
        StableDiffusionInpaintPipeline: 인페인팅 파이프라인 인스턴스
        
    Note:
        @st.cache_resource 데코레이터를 사용하여 모델을 메모리에 캐싱
    """
    print("Stable Diffusion Inpaint setup!")
    device = get_device()
    
    # Stable Diffusion 인페인팅 모델 로드
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",  # 모델 ID
        torch_dtype=torch.float16,  # 16비트 부동소수점 사용 (메모리 효율)
    ).to(device)
    
    return pipe

@st.cache_resource
def get_lama_cleaner():
    """
    LaMa Cleaner 모델을 로드하고 캐싱하는 함수
    
    Returns:
        torch.jit.ScriptModule: 로드된 LaMa 모델
        
    Note:
        @st.cache_resource 데코레이터를 사용하여 모델을 메모리에 캐싱
    """
    print("lama cleaner setup!")
    device = get_device()
    
    # LaMa 모델 URL과 MD5 해시값 설정
    LAMA_MODEL_URL = os.environ.get(
        "LAMA_MODEL_URL",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    )
    LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

    # LaMa 모델 로드 및 평가 모드로 설정
    lama_model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    return lama_model

@st.cache_resource
def get_instruct_pix2pix():
    """
    Instruct Pix2Pix 모델을 로드하고 캐싱하는 함수
    
    Returns:
        StableDiffusionInstructPix2PixPipeline: Pix2Pix 파이프라인 인스턴스
        
    Note:
        @st.cache_resource 데코레이터를 사용하여 모델을 메모리에 캐싱
    """
    print("Instruct Pix2Pix setup!")
    device = get_device()
    
    # Instruct Pix2Pix 모델 로드
    model_name = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # 16비트 부동소수점 사용
        safety_checker=None  # 안전성 검사 비활성화
    ).to(device)
    
    # 스케줄러 설정
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe