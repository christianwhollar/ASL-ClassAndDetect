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
    
    def __init__(self, train_data, trainloader, test_data, test_loader, model = ''):
        self.train_data = train_data
        self.train_loader = trainloader
        self.test_data = test_data
        self.test_loader = test_loader
        
        ssl._create_default_https_context = ssl._create_unverified_context
        
        if model:
            filename = os.getcwd() + model
            self.model = CPU_Unpickler(open(filename, 'rb')).load()
            
        else:
            self.model = models.mobilenet_v2(pretrained=True)
            
    def setup(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                nn.Linear(in_features=1280, out_features=29, bias=True),
                                nn.LogSoftmax(dim=1))
        
        for p in self.model.features[-3:].parameters():
            p.requires_grad = True  
            
        # choose your loss function
        self.criterion = nn.NLLLoss()

        # define optimizer to train only the classifier and the previous three block.
        self.optimizer = optim.Adam([{'params':self.model.features[-1].parameters()},
                                {'params':self.model.features[-2].parameters()},
                                {'params':self.model.features[-3].parameters()},
                                {'params':self.model.classifier.parameters()}], lr=0.0005)

        # define Learning Rate scheduler to decrease the learning rate by multiplying it by 0.1 after each epoch on the data.
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
    
    def train(self):
        epochs = 2
        step = 0
        steps = math.ceil(len(self.train_data)/(self.train_loader.batch_size))

        running_loss = 0
        print_every = 20
        trainlossarr=[]
        testlossarr=[]
        oldacc=0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for epoch in range(epochs):
            print(Style.RESET_ALL)
            print(f"--------------------------------- START OF EPOCH [ {epoch+1} ] >>> LR =  {self.optimizer.param_groups[-1]['lr']} ---------------------------------\n")
            
            for inputs, labels in tqdm(self.train_loader,desc=Fore.GREEN +f"* PROGRESS IN EPOCH {epoch+1} ",file=sys.stdout):
                self.model.train()
                step += 1
                inputs=inputs.to(device)
                labels=labels.to(device)

                self.optimizer.zero_grad()

                props = self.model.forward(inputs)
                loss = self.criterion(props, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (step % print_every == 0) or (step==steps):
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            props = self.model.forward(inputs)
                            batch_loss = self.criterion(props, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(props)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()     

                    tqdm.write(f"Epoch ({epoch+1} of {epochs}) ... "
                        f"Step  ({step:3d} of {steps}) ... "
                        f"Train loss: {running_loss/print_every:.3f} ... "
                        f"Test loss: {test_loss/len(self.test_loader):.3f} ... "
                        f"Test accuracy: {accuracy/len(self.test_loader):.3f} ")
                    
                    trainlossarr.append(running_loss/print_every)
                    testlossarr.append(test_loss/len(self.test_loader))
                    running_loss = 0        
                
            self.scheduler.step()
            step=0
    
    def test(self):
        print('Entering Test...')
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_loader = self.test_loader
        
        # Turn autograd off
        with torch.no_grad():

            # Set the model to evaluation mode
            model.eval()

            # Set up lists to store true and predicted values
            y_true = []
            test_preds = []

            # Calculate the predictions on the test set and add to list
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # Feed inputs through model to get raw scores
                logits = model.forward(inputs)
                # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
                probs = F.softmax(logits,dim=1)
                # Get discrete predictions using argmax
                preds = np.argmax(probs.cpu().numpy(),axis=1)
                # Add predictions and actuals to lists
                test_preds.extend(preds)
                y_true.extend(labels.cpu())

            # Calculate the accuracy
            test_preds = np.array(test_preds)
            y_true = np.array(y_true)
            test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
            
            # Recall for each class
            recall_vals = []
            for i in range(10):
                class_idx = np.argwhere(y_true==i)
                total = len(class_idx)
                correct = np.sum(test_preds[class_idx]==i)
                recall = correct / total
                recall_vals.append(recall)
        return test_acc,recall_vals
    
    def export(self):        
        outfile = os.getcwd() + '/models/mobilenetv2_asl.pkl'
        
        with open(outfile,'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)