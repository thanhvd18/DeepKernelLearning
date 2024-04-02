import os
import matplotlib.pyplot as plt
from utils import *
from sklearn.datasets import load_iris 
import pandas as pd

def load_data(dataPath, input=True):
    if input:
        data = pd.read_csv(dataPath,header=None)
    else:
        data = pd.read_csv(dataPath)    
    data = data.values()
    return data

def test_supervied_kernel(X=None,y=None, dataPathX=None,dataPathy=None):
    if X is None and y is None:
        iris = load_iris()
        X = iris.data
        y = iris.target
    elif dataPathX is not None:
        X = load_data(dataPathX,input=True)
        y = load_data(dataPathy)

    GM_kernel,GM_train_kernel, GM_val_kernel,GM_test_kernel,GM_Z = supervised_kernel(X,y)
    gt_kernel = gt_kernel(y)
    plt.figure()
    plt.imshow(GM_kernel)
    plt.savefig('./ignore/test_supervised_kernel.png')

    plt.figure()
    plt.imshow(gt_kernel)
    plt.savefig('./ignore/test_gt_kernel.png')