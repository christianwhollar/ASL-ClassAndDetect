from data.raw.download_data import *

class MakeDataset():
    
    def __init__(self):
        self.DownloadData = DownloadData()
        self.KaggleStatus = False
        self.RoboFlowStatus = False
        
        self.update_dataset_status()
        self.dataset_download()
        self.move_data()
        
    def update_dataset_status(self):
        proc_files= os.listdir(os.getcwd() + '/data/processed')
        kaggle_sub = 'asl'
        kaggle_empty = not any(kaggle_sub in f for f in proc_files)
        
        if kaggle_empty:
            self.KaggleStatus = True
        
    def dataset_download(self):
        if self.KaggleStatus:
            print('Kaggle Dataset Downloading...')
            self.DownloadData.download_kaggle()
            print('Kaggle Dataset Download Complete!')
            
    def move_data(self):
        self.DownloadData.move_data()
