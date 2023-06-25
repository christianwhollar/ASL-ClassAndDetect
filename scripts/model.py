from scripts.cpu_unpickler import CPU_Unpickler
from colorama import Fore, Style
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import ssl
from tqdm import tqdm
import math
import numpy as np
import os
import pickle

class ModelSetup():
    '''
    Generic PyTorch Model Class
    4 Parts: Setup, Train, Test, & Export
    '''
    
    def __init__(self, train_data, trainloader, test_data, test_loader, model = ''):
        '''
        Setup Train/Test Data & Loaders
        Import Model From TorchVision
        Args:
            None
        Returns:
            None
        '''
        self.train_data = train_data
        self.train_loader = trainloader
        self.test_data = test_data
        self.test_loader = test_loader
        
        ssl._create_default_https_context = ssl._create_unverified_context
        
        if model:
            filename = os.getcwd() + model
            self.model = CPU_Unpickler(open(filename, 'rb')).load()
            
        else:
            print('New Model Import')
            self.model = models.resnet18(pretrained=True)
            
    def setup(self):
        '''
        Setup Model Structure, Loss Function, Optimizer, LR Scheduler
        Args:
            None
        Returns:
            None
        '''
        
        # Require Grad to False for Raw Model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Structure
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                nn.Linear(in_features=1280, out_features=39, bias=True),
                                nn.LogSoftmax(dim=1))
        
        # Require Grad to True for New Layers
        for p in self.model.features[-3:].parameters():
            p.requires_grad = True  
            
        # Loss Func
        self.criterion = nn.NLLLoss()

        # Adam Optimizer
        self.optimizer = optim.Adam([{'params':self.model.features[-1].parameters()},
                                {'params':self.model.features[-2].parameters()},
                                {'params':self.model.features[-3].parameters()},
                                {'params':self.model.classifier.parameters()}], lr=0.0005)

        # Learning Rate Scheduler 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
    
    def train(self):
        '''
        Train Input Model via PyTorch
        Args:
            None
        Returns:
            None
        '''
        
        # Loop Params
        epochs = 2
        step = 0
        steps = math.ceil(len(self.train_data)/(self.train_loader.batch_size))
        running_loss = 0
        print_every = 20
        
        # Loss Lists
        trainlossarr=[]
        testlossarr=[]

        # Device Setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Epochs
        for epoch in range(epochs):
            # Status Print
            print(Style.RESET_ALL)
            print(f"--------------------------------- START OF EPOCH [ {epoch+1} ] >>> LR =  {self.optimizer.param_groups[-1]['lr']} ---------------------------------\n")
            
            # For Input, Class in Train Loader
            for inputs, labels in tqdm(self.train_loader,desc=Fore.GREEN +f"* PROGRESS IN EPOCH {epoch+1} ",file=sys.stdout):
                self.model.train()
                step += 1
                inputs=inputs.to(device)
                labels=labels.to(device)

                # Pytorch Movement
                self.optimizer.zero_grad()
                props = self.model.forward(inputs)
                loss = self.criterion(props, labels)
                loss.backward()
                self.optimizer.step()

                # Inc by Loss
                running_loss += loss.item()

                if (step % print_every == 0) or (step==steps):
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        # Test Loader Iteration
                        for inputs, labels in self.test_loader:
                            
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            # PyTroch Movement
                            props = self.model.forward(inputs)
                            batch_loss = self.criterion(props, labels)
                            test_loss += batch_loss.item()

                            # Accuracy
                            ps = torch.exp(props)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()    
                             
                    # Print Step, Train, Test Loss/Acc Update
                    tqdm.write(f"Epoch ({epoch+1} of {epochs}) ... "
                        f"Step  ({step:3d} of {steps}) ... "
                        f"Train loss: {running_loss/print_every:.3f} ... "
                        f"Test loss: {test_loss/len(self.test_loader):.3f} ... "
                        f"Test accuracy: {accuracy/len(self.test_loader):.3f} ")
                    
                    # Update Lists
                    trainlossarr.append(running_loss/print_every)
                    testlossarr.append(test_loss/len(self.test_loader))
                    running_loss = 0        
                
            self.scheduler.step()
            step=0
    
    def test(self):
        '''
        Calculate Preds/Acts for Test Data
        Acc, Recall Values
        Args:
            None
        Returns:
            None
        '''
        print('Entering Test...')
        
        # Setup Class Reqs, Device Setup
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_loader = self.test_loader
        
        # Classes From Test Data
        classes=self.test_data.class_to_idx

        # Turn Autograd Off
        with torch.no_grad():

            # Model to Eval Mode
            model.eval()

            # Lists for Acts, Preds
            y_true = []
            test_preds = []

            # Calculate Preds, Add Pred/Actual to List
            test_count = 1
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # Get Raw Scores
                logits = model.forward(inputs)
                
                # Raw Scores to Probabilities
                probs = F.softmax(logits,dim=1)
                
                # Preds From Probs
                preds = np.argmax(probs.cpu().numpy(),axis=1)
                
                # Extend Predictions/Actuals to List
                test_preds.extend(preds)
                y_true.extend(labels.cpu())
                
                # Print Test Update Per Iteration
                print('Test ' + str(test_count) + ' completed! ' + str(len(test_loader) - test_count) + ' tests remaining...')
                test_count += 1
                
            # Calculate Acc
            test_preds = np.array(test_preds)
            y_true = np.array(y_true)
            test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
            
            # Calculate Recall for Each Class
            recall_vals = []
            for i in range(len(classes)):
                class_idx = np.argwhere(y_true==i)
                total = len(class_idx)
                correct = np.sum(test_preds[class_idx]==i)
                recall = correct / total
                recall_vals.append(recall)
                
        # Print Test Acc
        print('Test set accuracy is {:.3f}'.format(test_acc))
        
        # Print Recall Values for Each Class
        for c, idx in classes.items():
            print('For class {}, recall is {}'.format(c,recall_vals[idx]))
            
        self.test_acc = test_acc
        self.recall_vals = recall_vals
        return test_acc, recall
    
    def export(self):        
        '''
        Export Trained Model to .pkl Format
        Args:
            None
        Returns:
            None
        '''
        outfile = os.getcwd() + '/models/resnet_asl_letnum.pkl'
        
        with open(outfile,'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)