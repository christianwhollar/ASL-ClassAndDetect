import io
import json
import pickle
from PIL import Image
import os
from scripts.cpu_unpickler import *
import streamlit as st
import torch
from torchvision import transforms, datasets
        
st.set_page_config(layout='wide')

st.title('Learning American Sign Language (ASL)')
st.subheader('Powered by AI')

mode = st.select_slider('', options = ['UPLOAD IMAGE', 'LIVE'])

if mode == 'UPLOAD IMAGE':
    # Load Model
    with open('classes.json', 'r') as f:
        classes = json.load(f)

    filename = os.getcwd() + '/models/mobilenetv2_asl.pkl'
    model = CPU_Unpickler(open(filename, 'rb')).load()
    
    upload_image = st.file_uploader('Upload your image here')
    
    if upload_image:
        img = Image.open(upload_image)
        
        test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        img_normalized = test_transforms(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        
        with torch.no_grad():
            model.eval()  
            output =model(img_normalized)
            # print(output)
            index = output.data.cpu().numpy().argmax()
            print(index)
            class_name = classes[str(index)]
            
        with st.columns(3)[1]:
            st.image(img,caption='Powered by MobileNetV2')
            st.markdown('The predicted class is ' + class_name + '!')