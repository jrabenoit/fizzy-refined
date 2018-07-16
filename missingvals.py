#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

#http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values

def Impute():
    #For sparse: Impute missing vals by most_frequent, scales as sparse matrix
    #For denser: Impute by median, scale by standard scaler
    info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/combined-study/input_data_after_dropna.csv', encoding='utf-8').set_index('PATIENT')
    
    #first, remove columns with NaN values
    info= info.dropna(axis='columns', how='all')
    
    #to get specific dataset:
    del labels['HAMD 1=REMIT'] 
    info=labels.join(info)
    
    #tested axes- we want axis 0 to impute down a column
    #Using mode because of sparsity- if 99% of values are 0, most likely that this will be a 0 too.
    X= Imputer().fit_transform(info)
    #Can use median imputation to protect against long tail features
    X= Imputer(strategy='median', axis=0).fit_transform(df2)

    #using a sparse data scaler due to the number of zeros from binarized variables
    ss= StandardScaler()
    mas= MaxAbsScaler()
    mms= MinMaxScaler() #Gets features to a 0-1 range
    rs=RobustScaler() #Less sensitive to outliers

    X2= mms.fit_transform(X)
    X3=pd.DataFrame(data=X2, columns=data.columns, index=data.index)
    
    labels=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/combined-study/labels_after_dropna.csv', encoding='utf-8').set_index('PATIENT')

    labels= pd.DataFrame(index=labels.index, data=labels['GROUPLABEL'], columns=['GROUPLABEL'])
    labels.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/class-labels.csv', index_label='PATIENT')
    
    
    #str((np.count_nonzero(X)/(X.shape[0]*X.shape[1]))*100) to get density... 96% sparse 
    #Should return to the psych scales (madrs, hamd etc) and encode using OneHotEncoder, since continuous variables are expected.
    
    #np.savetxt('/media/james/ext4data1/current/projects/pfizer/refined-combined-study/data.csv', X2, delimiter=',')
    X3.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/data.csv', index_label='PATIENT')
    
    return
