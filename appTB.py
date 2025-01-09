import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from io import BytesIO

def app():
    st.header('Med AI Tuberculosis Bacilli Detection')
    try:
        # Load the YOLOv8 model
        model = YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return

    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload photo", type=['jpeg', 'jpg', 'png'])
        selected_objects = st.multiselect('Choose objects to detect', object_names)
        min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)
        submitted = st.form_submit_button(label='Submit')

    if submitted:
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                image = Image.open(BytesIO(uploaded_file.read()))
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                with st.spinner('Processing image...'):
                    results = model(image_np)
                    for detection in results[0].boxes.data:
                        x0, y0, x1, y1, score, cls = detection[:6].tolist()
                        score = round(float(score), 2)
                        cls = int(cls)
                        
                        if score >= min_confidence and (not selected_objects or model.names[cls] in selected_objects):
                            cv2.rectangle(image_np, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                            label = f'{model.names[cls]}: {score}'
                            cv2.putText(image_np, label, (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    st.image(image_np, caption='Processed Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.warning("Please upload an image to process.")

if __name__ == "__main__":
    app()
