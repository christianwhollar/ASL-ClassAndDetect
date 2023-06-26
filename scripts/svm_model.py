import joblib 
import json
import numpy as np
import os
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score

class SVMModel():
    '''
    SVM Model Class for Image Classification
    4 Parts: Load, Train, Test, Setup
    Optimal params found using GridSearchCV
    '''
    def __init__(self):
        '''
        Stores Class Dictionary, Data Directory
        Returns:
            None
        Args:
            None
        '''
        with open('classes.json', 'r') as f:
            self.classes = json.load(f)
        self.datadir = os.getcwd() + '/data/processed/train'
        
    def load(self):
        '''
        Load in data to Pandas DataFrame.
        Split into test/train data.
        Set processed data to self.
        Args:
            None
        Returns:
            None
        '''
        flat_data_arr = []
        target_arr = []
        N=10
        
        # For class & idx, generate label/target data w/ image loads        
        for key, value in self.classes.items():
            
            print(f'loading... category : {key}')
            path=os.path.join(self.datadir,key)
            i = 0
            
            for img in os.listdir(path):
                if i >= N:
                    break
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(value)
                i+=1
                
        flat_data=np.array(flat_data_arr)
        target=np.array(target_arr)
        
        # Send to DataFrame, Extract x & y vals
        df=pd.DataFrame(flat_data) 
        df['Target']=target
        x=df.iloc[:,:-1] 
        y=df.iloc[:,-1]
        
        # Split Data
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=0.20,
                                               random_state=0,
                                               stratify=y)
        
    def train(self):
        '''
        Train SVM Model
        GridSearchCV for optimal parameters
        Args:
            None
        Returns:
            None
        '''
        # Setup Param Dict for GridSearchCV
        param_grid={'C':[0.1,1,10,100],
            'gamma':[0.0001,0.001,0.1,1],
            'kernel':['rbf','poly']}
  
        svc=svm.SVC(probability=True)
        self.model=GridSearchCV(svc,param_grid)
        
        # Fit model
        self.model.fit(self.x_train,self.y_train)
        
    def test(self):
        ''''
        Get Predictions For Test Data
        Calculate accuracy
        Args:
            None
        Returns:
            None
        '''
        y_pred = self.model.predict(self.x_test)
        accuracy = self.accuracy_score(y_pred, self.y_test)
        print(f"The model is {accuracy*100}% accurate")
        recall_vals_svm = recall_score(y_true = self.y_test, y_pred = y_pred)
        
        recall_vals_svm_json = json.dumps(recall_vals_svm)
        with open(os.getcwd() + "/data/output/recall_svm.json", "w") as outfile:
            outfile.write(recall_vals_svm_json)
            
    def export(self):
        '''
        Dump Trained Model to .pkl File
        Args:
            None
        Returns:
            None
        '''
        joblib.dump(self.model.best_estimator_, os.getcwd() + '/models/svm_model.pkl', compress = 1)