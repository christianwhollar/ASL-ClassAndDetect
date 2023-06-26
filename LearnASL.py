import io
import joblib
import json
import pickle
from PIL import Image
import numpy as np
import os
import pandas as pd
from scripts.cpu_unpickler import *
from skimage.io import imread
from skimage.transform import resize
import streamlit as st
import torch
from torchvision import transforms, datasets
        
st.set_page_config(layout='wide')

st.title('Learning American Sign Language (ASL)')
st.subheader('Powered by AI')

mode = st.select_slider('', options = ['MobileNetV2', 'SupportVectorMachine'])

if mode == 'MobileNetV2':
    # Load Model
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    
    filename = os.getcwd() + '/models/mobilenetv2_asl_letnum.pkl'
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
            key_list = list(classes.keys())
            val_list = list(classes.values())
            position = val_list.index(index)
            class_name = key_list[position]

        with st.columns(3)[1]:
            st.image(img,caption='Powered by MobileNetV2')
            st.markdown('The predicted class is ' + class_name + '!')
            
elif mode == 'SupportVectorMachine':
    # Load Model
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    
    filename = os.getcwd() + '/models/svm_model.pkl'
    model = joblib.load(filename)
    
    upload_image = st.file_uploader('Upload your image here')
    
    if upload_image:
        img = Image.open(upload_image)
        flat_data_arr = []
        target_arr = []
        img_array=imread('A1.jpg')
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(0)
                        
        flat_data=np.array(flat_data_arr)
        target=np.array(target_arr)

        # Send to DataFrame, Extract x & y vals
        df=pd.DataFrame(flat_data) 
        df['Target']=target
        x=df.iloc[:,:-1] 
        y=df.iloc[:,-1]
        index = model.predict(x)
        
        key_list = list(classes.keys())
        val_list = list(classes.values())
        position = val_list.index(index[0])
        class_name = key_list[position]
        
        with st.columns(3)[1]:
            st.image(img,caption='Powered by MobileNetV2')
            st.markdown('The predicted class is ' + class_name + '!')