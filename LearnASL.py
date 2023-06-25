import io
import pickle
import os
import streamlit as st
import torch

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

mode = st.select_slider('Select Mode', options = ['UPLOAD IMAGE', 'LIVE'])

if mode == 'UPLOAD IMAGE':
    # Load Model
    filename = os.getcwd() + '/models/mobilenetv2_asl.pkl'
    model = CPU_Unpickler(open(filename, 'rb')).load()
    
    upload_image = st.file_uploader('Upload your image here')
    
    if upload_image:
        print('Image Uploaded')
    
elif mode == 'LIVE':
    filename = os.getcwd() + '/models/mobilenetv2_asl.pkl'
    model = CPU_Unpickler(open(filename, 'rb')).load()