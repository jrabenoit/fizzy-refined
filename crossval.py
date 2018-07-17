#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def OuterCv():   
    a=input('Click and drag DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag labels file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')    
    
    X= data
    y= np.array(labels[labels.columns[0]])
    
    X_train, X_test, y_train, y_test= [],[],[],[]
    outer_cv= {'X_train': [], 
               'X_test': [], 
               'y_train': [], 
               'y_test': []}

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X,y):
        X_train, X_test= X.index[train_index], X.index[test_index]
        y_train, y_test= y[train_index], y[test_index]
        
        outer_cv['X_train'].append(X_train)
        outer_cv['X_test'].append(X_test)
        outer_cv['y_train'].append(y_train)
        outer_cv['y_test'].append(y_test)
    
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/ocv.pickle', 'wb') as f: pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    #Set up as a flat structure of 25 folds, 5 from each training fold
    a=input('Click and drag OUTER CV file here: ')
    a=a.strip('\' ')
    with open(a, 'rb') as f: outer_cv= pickle.load(f)
    
    b=input('Click and drag labels file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')  
  
    X= outer_cv['X_train']
    y= outer_cv['y_train']
  
    X_train, X_test, y_train, y_test= [],[],[],[]
    inner_cv= {'X_train': [], 
               'X_test': [], 
               'y_train': [], 
               'y_test': []}
    
    for X_, y_ in zip(X, y): 
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_,y_):      
            X_train, X_test= X_[train_index], X_[test_index]
            y_train, y_test= y_[train_index], y_[test_index]
            
            inner_cv['X_train'].append(X_train)
            inner_cv['X_test'].append(X_test)
            inner_cv['y_train'].append(y_train)
            inner_cv['y_test'].append(y_test)

    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/icv.pickle', 'wb') as f: pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return
