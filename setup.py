from scripts.make_dataset import *
from scripts.build_features import *
from scripts.model import *

if __name__ == '__main__':
    MakeDataset()
    
    bf = BuildFeatures(
        batch_size = 512,
        train_path = './data/processed/asl_alphabet_train',
        valid_path = './data/processed/asl_alphabet_valid',
        test_path = './data/processed/asl_alphabet_valid' 
    )
    
    bf.build_transformers()
    train_data, test_data = bf.build_data()
    trainloader, testloader = bf.build_dataloaders()
    
    ms = ModelSetup(train_data, trainloader, test_data, testloader)
    ms.setup()
    ms.train()
    ms.export()