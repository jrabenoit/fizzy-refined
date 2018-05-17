#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def OuterCv():   
    a=input('Click and drag data file here: ')
    a=a.strip('\' ')
    data=np.loadtxt(a, delimiter=',')
    b=input('Click and drag labels file here: ')
    b=b.strip('\' ')
    
    c=input('Click and drag holdout data file here: ')
    c=c.strip('\' ')
    d=input('Click and drag holdout labels file here: ')
    d=d.strip('\' ')
    
    
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')    
    labels=np.array(labels[labels.columns[0]])

    holdout_labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')    
    holdout_labels=np.array(labels[labels.columns[0]])
    
    outer_cv= {'X_train': [], 'X_test': [], 
               'y_train': [], 'y_test': [],
               'train_indices': [], 'test_indices':[]}

    X= data
    y= labels
    
    X_train, X_test, y_train, y_test= [], [], [], []      

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X,y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]       
        
        outer_cv['X_train'].append(X_train)
        outer_cv['X_test'].append(X_test)
        outer_cv['y_train'].append(y_train)
        outer_cv['y_test'].append(y_test)
        outer_cv['train_indices'].append(train_index)
        outer_cv['test_indices'].append(test_index)
    
    with open('/media/james/ext4data1/current/projects/pfizer/refined-combined-study/ocv.pickle', 'wb') as f: pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    '''Set up as a flat structure of 25 df'''
    with open('/media/james/ext4data1/current/projects/pfizer/refined-combined-study/ocv.pickle', 'rb') as f: outer_cv= pickle.load(f)

    inner_cv= {'X_train': [], 'X_test': [], 
               'y_train': [], 'y_test': [],
               'train_indices': [], 'test_indices':[]}
    
    X= outer_cv['X_train']
    y= outer_cv['y_train']
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    #change X to subjects, y to labels
    #read loop as, "for each pair of X and y lists in (X,y)"
    
    for X_, y_ in zip(X, y): 
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_,y_):      
            X_train, X_test= X_[train_index], X_[test_index]
            y_train, y_test= y_[train_index], y_[test_index]

            inner_cv['X_train'].append(X_train)
            inner_cv['X_test'].append(X_test)
            inner_cv['y_train'].append(y_train)
            inner_cv['y_test'].append(y_test)
            inner_cv['train_indices'].append(train_index)
            inner_cv['test_indices'].append(test_index)

    with open('/media/james/ext4data1/current/projects/pfizer/refined-combined-study/icv.pickle', 'wb') as f: pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return

    
