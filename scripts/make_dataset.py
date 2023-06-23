import sys
import os
 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from constants import *
from roboflow import Roboflow

def download_ASL_robowflow(raw_dir):
    rf = Roboflow(api_key=ROBOFLOW_KEY, model_format="yolov7")
    rf.workspace().project(ROBOWFLOW_PROJECT).version(ROBOWFLOW_VERSION).download(location=raw_dir)