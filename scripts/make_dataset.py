from data.raw.download_data import *

class MakeDataset():
    '''
    Orchestrate (/data/raw/) DownloadData class downloading methods
    '''
    def __init__(self):
        '''
        Setup DownloadData class
        Call MakeDataset class methods
        Args:
            None
        Returns:
            None
        '''
        self.DownloadData = DownloadData()
        self.KaggleStatus = False
        
        self.update_dataset_status()
        self.dataset_download()
        self.move_data()
        
    def update_dataset_status(self):
        '''
        Set KaggleStatus: kaggle data download required
        Args:
            None
        Returns:
            None
        '''
        proc_files= os.listdir(os.getcwd() + '/data/processed')
        kaggle_sub = 'asl'
        kaggle_empty = not any(kaggle_sub in f for f in proc_files)
        
        if kaggle_empty:
            self.KaggleStatus = True
        
    def dataset_download(self):
        '''
        Check if data downloaded, download if not
        Args:
            None
        Returns:
            None
        '''
        if self.KaggleStatus:
            print('Kaggle Dataset Downloading...')
            self.DownloadData.download_kaggle()
            print('Kaggle Dataset Download Complete!')
            
    def move_data(self):
        '''
        Args:
            None
        Returns:
            None
        '''
        self.DownloadData.move_data()
