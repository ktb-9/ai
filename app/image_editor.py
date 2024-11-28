# 필요한 라이브러리 임포트
import streamlit as st
from st_pages import add_page_title
from streamlit_drawable_canvas import st_canvas  # 그리기 가능한 캔버스 컴포넌트
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
import time

# 사용자 정의 유틸리티 함수들 임포트
from utils.util import resize_image, xywh2xyxy
from utils.template import image_editor_template
from utils.agent import image_editor_agent
from utils.action import backward_inference_image, forward_inference_image, reset_inference_image, reset_coord
from utils.inference import sam

def image_editor():
    """
    메인 이미지 편집기 함수
    - 이미지 업로드, 편집, 세그멘테이션 기능 제공
    - 사용자 상호작용을 위한 UI 구성
    """
    add_page_title()
    
    # 지원하는 기능 표시
    st.caption("기능: zero-shot segmentation, image2image, inpaint, erase", unsafe_allow_html=True)

    # 이미지 업로드 인터페이스
    uploaded_image = st.file_uploader("Upload a your image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:  
        # 채팅 입력 필드 생성
        prompt = st.chat_input("Send a message")
        
        # 세션 상태 초기화 (처음 이미지가 업로드될 때)
        if "image_state" not in st.session_state:
            try:
                # 이미지 로드 및 전처리
                image = Image.open(uploaded_image).convert('RGB')
                image = ImageOps.exif_transpose(image)  # 이미지 자동회전 방지
                resized_image, resize_ratio = resize_image(image, max_width=704, max_height=704)
                
                # 세션 상태 초기화
                st.session_state["image_state"] = 0  # 현재 이미지 상태 인덱스
                st.session_state["inference_image"] = [resized_image]  # 처리된 이미지 히스토리
                st.session_state["sam_image"] = None  # 세그멘테이션 결과 이미지
                st.session_state["num_coord"] = 0  # 좌표 개수
            except Exception as e:
                st.error(f"이미지 로딩 에러: {str(e)}")
                return
        
        # 캔버스 상태 관리
        if "canvas" in st.session_state and st.session_state["canvas"] is not None:  
            try:
                if isinstance(st.session_state["canvas"], dict) and "objects" in st.session_state["canvas"]:
                    df = pd.json_normalize(st.session_state["canvas"]["objects"])
                    # 세그멘테이션 이미지가 있고 캔버스가 비어있으면 상태 초기화
                    if st.session_state["sam_image"] is not None and len(df) == 0:
                        st.session_state["num_coord"] = 0
                        st.session_state["sam_image"] = None
            except Exception as e:
                st.error(f"캔버스 처리 에러: {str(e)}")
        
        # 드로잉 도구 선택 UI
        drawing_mode = st.selectbox("Drawing tool:", ("rect", "point", "freedraw"), on_change=reset_coord)
        
        # 드로잉 도구별 설정
        if drawing_mode == "freedraw":
            col1, col2 = st.columns(2)
            anno_color = col1.color_picker("Annotation color: ", "#141412") + "77"  # 알파값 추가
            brush_width = col2.number_input("Brush width", value=40)
        else:
            anno_color = st.color_picker("Annotation color: ", "#141412") + "77"
        
        # 캔버스 표시 영역
        col1, col2 = st.columns((0.1,1))
        with col2:   
            try:
                # 현재 표시할 이미지 선택 (세그멘테이션 결과 또는 현재 상태 이미지)
                display_image = (st.session_state["sam_image"] 
                            if st.session_state.get("num_coord", 0) != 0 
                            else st.session_state["inference_image"][st.session_state["image_state"]])
            
                # 캔버스 컴포넌트 생성
                canvas = st_canvas(
                    fill_color=anno_color,
                    stroke_width=brush_width if drawing_mode == "freedraw" else 2,
                    stroke_color="black" if drawing_mode != "freedraw" else anno_color,
                    background_image=display_image,
                    height=st.session_state["inference_image"][0].height,
                    width=st.session_state["inference_image"][0].width,
                    drawing_mode=drawing_mode,
                    key=f"canvas_{st.session_state.get('canvas_key', 0)}",  # 동적 캔버스 키
                    point_display_radius=4
                )
            except Exception as e:
                st.error(f"캔버스 생성 에러: {str(e)}")
                return
        
        # 제어 버튼 영역
        col1, col2, _, col3, col4 = st.columns((4,4,10,3,4))
        col1.button("backward", on_click=backward_inference_image, use_container_width=True)
        col2.button("forward", on_click=forward_inference_image, use_container_width=True)
        col3.button("reset", on_click=reset_inference_image, use_container_width=True)
        
        # 이미지 다운로드 버튼
        try:
            buf = BytesIO()
            st.session_state["inference_image"][st.session_state["image_state"]].save(buf, format="JPEG")
            byte_im = buf.getvalue()

            col4.download_button(
                label="Download",
                data=byte_im,
                file_name=uploaded_image.name,
                mime="image/jpeg",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"다운로드 버튼 생성 에러: {str(e)}")
        
        # 캔버스 입력 처리
        if canvas is not None and canvas.json_data is not None and len(canvas.json_data.get("objects", [])) > 0:
            try:
                # 캔버스 객체 데이터 처리
                df = pd.json_normalize(canvas.json_data["objects"])
                if len(df) == 0:
                    st.session_state["coord"] = False
                else:    
                    st.session_state["coord"] = True

                # 새로운 드로잉 입력 감지 및 처리
                if len(df) != 0 and st.session_state["num_coord"] != len(df):
                    st.session_state["num_coord"] = len(df)
                    st.session_state["freedraw"] = False
                    
                    # 사각형 모드 처리
                    if drawing_mode == "rect":
                        pos_coords = xywh2xyxy(df[["left", "top", "width", "height"]].values)
                        neg_coords = np.zeros([1, 2])
                        labels = np.array([2, 3])  # box width, box height

                        # SAM 모델을 사용한 세그멘테이션
                        image, mask, segmented_image = sam(image=st.session_state["inference_image"][st.session_state["image_state"]],
                                                           pos_coords=pos_coords,
                                                           neg_coords=neg_coords,
                                                           labels=labels)
                        
                        st.session_state["sam_image"] = segmented_image
                        st.session_state["mask"] = mask
                        st.rerun()
                    
                    # 포인트 모드 처리    
                    elif drawing_mode == "point":
                        pos_coords = df[["left", "top"]].values
                        neg_coords = np.zeros([1, 2])
                        labels = np.array([1])  # point
                        
                        # SAM 모델을 사용한 세그멘테이션
                        image, mask, segmented_image = sam(image=st.session_state["inference_image"][st.session_state["image_state"]],
                                                           pos_coords=pos_coords,
                                                           neg_coords=neg_coords,
                                                           labels=labels)
                        
                        st.session_state["sam_image"] = segmented_image
                        st.session_state["mask"] = mask
                        st.rerun()
                    
                    # 자유 드로잉 모드 처리    
                    elif drawing_mode == "freedraw":
                        if canvas.image_data is not None:
                            st.session_state["mask"] = canvas.image_data[:, :, -1] > 0
                            st.session_state["freedraw"] = True
                        
            except Exception as e:
                st.error(f"캔버스 입력 처리 에러: {str(e)}")
        
        # 프롬프트 입력 처리
        if prompt:
            with st.spinner(text="Please wait..."):
                try:
                    # 이미지 편집 에이전트를 사용한 변환 처리
                    agent = image_editor_agent()                  
                    transform_pillow = agent(image_editor_template(prompt=prompt))['output']
                    
                    # 변환된 이미지를 히스토리에 추가
                    st.session_state["inference_image"].insert(st.session_state["image_state"]+1, transform_pillow)
                    st.session_state["image_state"] += 1
                    
                    # 캔버스 상태 초기화
                    st.session_state["num_coord"] = 0
                    st.session_state["sam_image"] = None
                    
                    # 캔버스 강제 리셋을 위한 키 변경
                    if "canvas_key" not in st.session_state:
                        st.session_state["canvas_key"] = 0
                    st.session_state["canvas_key"] += 1
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"프롬프트 처리 에러: {str(e)}")
    else:
        # 이미지가 업로드되지 않은 경우 세션 상태 초기화
        st.session_state.clear()               

# 메인 실행 부분
if __name__ == "__main__":
    image_editor()