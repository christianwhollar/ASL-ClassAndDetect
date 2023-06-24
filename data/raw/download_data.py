from constants import *
from kaggle.api.kaggle_api_extended import KaggleApi
import os

class DownloadData():
    
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_API_KEY
    
    def download_kaggle(self, DATASET_NAME = 'amarinderplasma/alphabets-sign-language', PATH_NAME = './data/processed/'):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_NAME, path = PATH_NAME, unzip=True)
        
    # def download_obowflow(raw_dir):
    #     rf = Roboflow(api_key=ROBOFLOW_KEY, model_format="yolov7")
    #     rf.workspace().project(ROBOWFLOW_PROJECT).version(ROBOWFLOW_VERSION).download(location=raw_dir)