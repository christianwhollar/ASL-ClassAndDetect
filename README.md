# ASL Classification Project Description
#### Christian Hollar



#### About the Data
2 Kaggle Datasets:
* 'amarinderplasma/alphabets-sign-language' : contains real ASL images for letters A - Z, delete, & space (.jpg)
* 'lexset/synthetic-asl-numbers' : contains GAN generated synthetic ASL images for numbers 1 - 10 (.png)

Summary: 38 Classes: ASL Letters (A-Z), ASL Numbers (1-10), Delete, Nothing, & Space

#### About the Models

#### Project Structure


#### Model Training and Evaluation
* **SVM (Non-Deep Learning Model)**
* **MobileNetv2 (Deep Learning Model)**

#### Repository Structure
```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── classes.json            <- .json file to store classes, idx in dict
├── constants.py            <- constants file for kaggle api storage
├── LearnASL.py             <- streamlit app file 
├── setup.py                <- script to set up project (get data, build features, train model)
├── main.ipynb              <- main script/notebook to run model, streamlit web application from colab
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data 
    ├── build_features.py   <- script to run pipeline to generate features
    ├── model.py            <- script to train deep learning model and predict
    ├── cpu_unpickler.py    <- script to load gpu pkl model file to cpu
    ├── svm_model.py        <- script to train non deep learning model and predict
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for script to download
    ├── processed           <- directory to store processed data
├── .gitignore              <- git ignore file
```