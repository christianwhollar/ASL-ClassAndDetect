import torch
from torchvision import transforms, datasets

class BuildFeatures():
    
    def __init__(self, batch_size = '', train_path = '', valid_path = '', test_path = ''):
        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
    
    def build_transformers(self):
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
        self.train_data = datasets.ImageFolder(self.train_path, transform = self.train_transforms)
        self.test_data = datasets.ImageFolder(self.valid_path, transform = self.test_transforms)

    def build_dataloaders(self):
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=512, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=512)