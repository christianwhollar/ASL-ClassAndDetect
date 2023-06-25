from data.raw.download_data import *

class MakeDataset():
    
    def __init__(self):
        self.DownloadData = DownloadData()
        self.KaggleStatus = False
        self.RoboFlowStatus = False
        
        self.update_dataset_status()
        self.dataset_download()
        
    def update_dataset_status(self):
        print(os.getcwd())
        if not os.listdir(os.getcwd() + '/data/raw/processed'):
            self.KaggleStatus = True
        
    def dataset_download(self):
        if self.KaggleStatus:
            print('Kaggle Dataset Downloading...')
            self.DownloadData.download_kaggle()
            print('Kaggle Dataset Download Complete!')
