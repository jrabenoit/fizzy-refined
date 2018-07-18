#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def OuterCv():   
    a=input('Click and drag DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')    
    
    X= data
    y= np.array(labels[labels.columns[0]])
    
    train, test= [],[]
    
    outer_cv= {'train': [], 
               'test': []}

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X,y):
        train= X.index[train_index]
        test= X.index[test_index]
        outer_cv['train'].append(train)
        outer_cv['test'].append(test)

    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/outer_cv.pickle', 'wb') as f: pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    a=input('Click and drag DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT')  
    
    a=input('Click and drag OUTER CV file here: ')
    a=a.strip('\' ')
    with open(a, 'rb') as f: outer_cv= pickle.load(f)
        
    for i in range(len(outer_cv['train'])): 
        subjects=pd.DataFrame(index=outer_cv['train'][i])
        X= subjects.join(data)
        y= subjects.join(labels)
    
        train, test= [],[]
        inner_cv= {'train': [], 
                   'test': []}
    
        skf = StratifiedKFold(n_splits=5)   
        for train_index, test_index in skf.split(X,y):      
            train= X.index[train_index]
            test= X.index[test_index]
            inner_cv['train'].append(train)
            inner_cv['test'].append(test)
            
        with open('/media/james/ext4data1/current/projects/pfizer/combined-study/inner_cv_fold_'+str(i+1)+'.pickle', 'wb') as f: pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return
