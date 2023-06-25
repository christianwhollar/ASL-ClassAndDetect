import torch
from torchvision import transforms, datasets

class BuildFeatures():
    '''
    Setup data before calling model
    3 Parts: Build Transformers, Build Data, & Build DataLoaders
    '''
    def __init__(self, batch_size = -1, train_path = '', valid_path = '', test_path = ''):
        '''
        Args:
            batch_size (int) : batch size
            train_path (str) : training data directory path
            valid_path (str) : valid data directory path
            test_path (str) : test data directory path
        Returns:
            None
        '''
        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
    
    def build_transformers(self):
        '''
        Build train, test data transforms
        Set to self
        Args:
            None
        Returns:
            None
        '''
        self.train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(p=0.3),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        
        self.test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        
    def build_data(self):
        '''
        Build train data, test data
        CV data stored as ImageFolder
        Args:
            None
        Returns:
            train_data (torchvision.datasets.ImageFolder)
            test_data (torchvision.datasets.ImageFolder)
        '''
        self.train_data = datasets.ImageFolder(self.train_path, transform = self.train_transforms)
        self.test_data = datasets.ImageFolder(self.valid_path, transform = self.test_transforms)
        return self.train_data, self.test_data
    
    def build_dataloaders(self):
        '''
        Build trainloader, testloader
        Args:
            None
        Returns:
            trainloader (torch.utils.data.DataLoader) : train dataloader
            testloader (torch.utils.data.DataLoader) : test dataloader
        '''
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=512 * 2, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=512 * 2)
        return self.trainloader, self.testloader