import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def predict(img, model, faces):
    mask_label = {0:'mask', 1:'no mask'}
    mask_color  = {0:(0,255,0), 1:(255,0,0)}

    if len(faces) >= 1:
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(224,224))
            crop = np.reshape(crop,[1,224,224,3])
            crop = preprocess_input(crop)
            mask_result = model.predict(crop)
            cv2.putText(img, mask_label[mask_result.argmax()], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, mask_color[mask_result.argmax()], 2)
            cv2.rectangle(img,(x,y),(x+w,y+h),mask_color[mask_result.argmax()],1)
        return img
    return None
    

def extract_faces(img, face_model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples
    return faces


def main():
    st.title('Face Mask Detection')
    st.text('Built with Streamlit and Tensorflow')
    st.write("This is a simple image classification web application to detect faces and check if they are wearing masks")

    file = st.file_uploader("Please upload an image", type = ["jpg", "png", "jpeg"])
    if file is not None:
        img = Image.open(file)
        st.text("Original Image")
        st.image(img)
    
    # load model
    model = load_model('./models/masknet.h5') # classifier 
    face_model = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml') # face detector
    
    # convert PIL to cv2 image
    new_img = np.array(img.convert('RGB'))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    # detect faces and classify
    faces = extract_faces(new_img, face_model)
    predict_image = predict(new_img, model, faces)
    predict_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    if predict_image is None:
        st.text("Couldn't detect any faces in the image")
    else:
        st.image(predict_image)
        st.text(f"Detected {len(faces)} faces in the image")
    
if __name__ == '__main__':
    main()