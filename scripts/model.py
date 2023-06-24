from colorama import Fore, Style
import sys
import torch
from torch import nn, optim
from torchvision import models
import tqdm
import math

class ModelSetup():
    
    def __init__(self, train_data, trainloader, test_data, test_loader):
        
        self.model = models.mobilenet_v2(pretrained=True)
        self.epochs = 2
        self.step = 0
        self.running_loss = 0
        self.print_every = 20
        self.trainlossarr=[]
        self.testlossarr=[]
        self.oldacc=0

        self.steps = math.ceil(len(train_data)/(trainloader.batch_size))

    def setup(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                nn.Linear(in_features=1280, out_features=29, bias=True),
                                nn.LogSoftmax(dim=1))
        
        for p in self.model.features[-3:].parameters():
            p.requires_grad = True  
            
        # choose your loss function
        criterion = nn.NLLLoss()

        # define optimizer to train only the classifier and the previous three block.
        self.optimizer = optim.Adam([{'params':self.model.features[-1].parameters()},
                                {'params':self.model.features[-2].parameters()},
                                {'params':self.model.features[-3].parameters()},
                                {'params':self.model.classifier.parameters()}], lr=0.0005)

        # define Learning Rate scheduler to decrease the learning rate by multiplying it by 0.1 after each epoch on the data.
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
    
    def train(self, trainloader, n_epochs):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for epoch in range(n_epochs):
            print(Style.RESET_ALL)
            print(f"--------------------------------- START OF EPOCH [ {epoch+1} ] >>> LR =  {self.optimizer.param_groups[-1]['lr']} ---------------------------------\n")
            
            for inputs, labels in tqdm(trainloader,desc=Fore.GREEN +f"* PROGRESS IN EPOCH {epoch+1} ",file=sys.stdout):
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

                if (step % self.print_every == 0) or (step==self.steps):
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            props = self.model.forward(inputs)
                            batch_loss = self.criterion(props, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(props)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()     

                    tqdm.write(f"Epoch ({epoch+1} of {self.epochs}) ... "
                        f"Step  ({step:3d} of {self.steps}) ... "
                        f"Train loss: {running_loss/self.print_every:.3f} ... "
                        f"Test loss: {test_loss/len(self.testloader):.3f} ... "
                        f"Test accuracy: {accuracy/len(self.testloader):.3f} ")
                    self.trainlossarr.append(running_loss/self.print_every)
                    self.testlossarr.append(test_loss/len(self.testloader))
                    running_loss = 0        
                
            self.scheduler.step()
            step=0

    
    def test(self):
        return
    
    def predict(self):
        return