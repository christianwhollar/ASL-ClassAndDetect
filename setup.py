from scripts.build_features import *
from scripts.make_dataset import *
from scripts.model import *
from scripts.svm_model import *

if __name__ == '__main__':
    # Make Dataset
    MakeDataset()
    
    # Build Features
    bf = BuildFeatures(
        batch_size = 512,
        train_path = './data/processed/train',
        valid_path = './data/processed/test',
        test_path = './data/processed/asl_alphabet_test'
    )
    
    # Transfer Learning Model - Data, DataLoaders
    bf.build_transformers()
    train_data, test_data = bf.build_data()
    trainloader, testloader = bf.build_dataloaders()
    
    # Transfer Learning - Setup, Train, Test, Export
    ms = ModelSetup(train_data, trainloader, test_data, testloader, model = '/models/mobilenetv2_asl_letnum.pkl')
    ms.setup()
    ms.train()
    test_acc, recall_vals = ms.test()
    ms.export()
    
    # SVM Model - Setup, Train, Test, Export
    # svm = SVMModel()
    # svm.load()
    # svm.train()
    # svm.test()
    # svm.export()
    