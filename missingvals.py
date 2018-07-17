#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

def Impute():
    info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/combined-study/final-combined-tables/data_before_fillna.csv', encoding='utf-8').set_index('PATIENT')
    labels=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/combined-study/labels.csv', encoding='utf-8').set_index('PATIENT')
    
    info= info.dropna(axis='columns', how='all')
    
    del labels[labels.columns[0]] 
    
    info=labels.join(info)
    X= Imputer().fit_transform(info)
    #X= Imputer(strategy='median', axis=0).fit_transform(info)

    mms= MinMaxScaler()
    X2= mms.fit_transform(X)
    X3=pd.DataFrame(data=X2, columns=info.columns, index=info.index)
    
    X3.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/data.csv', index_label='PATIENT')
    
    return
