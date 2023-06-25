# ASL Classification Project Description
#### Christian Hollar

38 Classes 

#### About the Data
* test1
* test2

#### Data Processing

#### Project Structure

#### Model Training and Evaluation
* **SVM (Non-Deep Learning Model)**
* **MobileNetv2 (Deep Learning Model)**

#### Repository Structure
```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile [OPTIONAL]     <- setup and run project from command line
├── setup.py                <- script to set up project (get data, build features, train model)
├── main.py [or main.ipynb] <- main script/notebook to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data [OPTIONAL]
    ├── build_features.py   <- script to run pipeline to generate features [OPTIONAL]
    ├── model.py            <- script to train model and predict [OPTIONAL]
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw data or script to download
    ├── processed           <- directory to store processed data
    ├── outputs             <- directory to store any output data
├── notebooks               <- directory to store any exploration notebooks used
├── .gitignore              <- git ignore file
```