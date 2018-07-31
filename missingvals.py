#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

def Impute():
    a=input('Click and drag DATASET WITH MISSING VALUES file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    data= data.dropna(axis='columns', how='all')
    
    del labels[labels.columns[0]] 
    
    data=labels.join(data)
    X= Imputer().fit_transform(data)
    
    mms= MinMaxScaler()
    X2= mms.fit_transform(X)
    X3=pd.DataFrame(data=X2, columns=data.columns, index=data.index)
    
    X3.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/data.csv', index_label='PATIENT')
    
    return
    
