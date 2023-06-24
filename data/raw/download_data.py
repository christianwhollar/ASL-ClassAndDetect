from constants import *
from kaggle.api.kaggle_api_extended import KaggleApi
import os

class download_data():
    
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_API_KEY
    
    def download_kaggle(self, DATASET_NAME = 'amarinderplasma/alphabets-sign-language', PATH_NAME = './data/processed/'):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_NAME, path = PATH_NAME, unzip=True)
        
if __name__ == '__main__':
    dd = download_data()
    dd.download_kaggle()