#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoLarsIC
import copy, pickle
import numpy as np
import pandas as pd
import itertools

#To flatten feature list and get frequencies: 
#[item for sublist in inner_cv['Feature Indices'] for item in sublist]
#from collections import Counter
#b=dict(Counter(a))

def InnerFeats():
    a=input('Click and drag DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag INNER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: inner_cv= pickle.load(f)
    
    folds= len(inner_cv['train'])
    feats=[[0]]*folds
    
    for i in range(folds):        
        subjects=pd.DataFrame(index=inner_cv['train'][i])
        X= subjects.join(data)
        y= subjects.join(labels)
              
        llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        llic.fit(X, y)
        feats[i]=llic.get_support(indices=True)    
    
    foldgroup=[]
    for i in range(0, len(feats), 5):
        foldgroup.append(feats[i:i+5])
    
    for i in range(len(foldgroup)):
        featlist= list(set.intersection(*map(set,foldgroup[i])))
        featlist.sort()
        
        data_cut= data[data.columns[featlist]]
        #going to get all subjects back out of this- split into train/test groups during estimators.
        
        data_cut.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/data-cut-to-feature-set-for-fold-'+str(i+1)+'.csv', index_label='PATIENT')
        
    return    

def OuterFeats():
    a=input('Click and drag DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag OUTER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: outer_cv= pickle.load(f)
    
    folds= len(outer_cv['train'])
    feats=[[0]]*folds
    
    for i in range(folds):
        subjects=pd.DataFrame(index=outer_cv['train'][i])
        X= subjects.join(data)
        y= subjects.join(labels)

        llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        llic.fit(X,y)
        feats[i]=llic.get_support(indices=True)
    
    featlist= list(set.intersection(*map(set,feats)))
    featlist.sort()
    feature_csv= pd.DataFrame(index=featlist, data= list(data.columns[featlist]))
    feature_csv.index.name='Feature #'
    feature_csv.columns=['Feature Name']

    print(len(featlist))
    
    data_cut= data[data.columns[featlist]]
     
    data_cut.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/data-cut-to-feature-set.csv', index_label='PATIENT')
    feature_csv.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/intersecting-features-index.csv')
       
    return
