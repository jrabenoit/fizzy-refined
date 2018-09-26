#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoLarsIC, LassoCV
import copy, pickle
import numpy as np
import pandas as pd
import itertools

#To flatten feature list and get frequencies: 
#[item for sublist in inner_cv['Feature Indices'] for item in sublist]
#from collections import Counter
#b=dict(Counter(a))

def OuterFeats():
    a=input('Click and drag ENTIRE DATASET file here: ')
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

        llic= SelectFromModel(LassoLarsIC())
        #llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        
        llic.fit(X,y)
        feats[i]=llic.get_support(indices=True)
    
    featlist= list(set.intersection(*map(set,feats)))
    featlist.sort()
    feature_csv= pd.DataFrame(index=featlist, data= list(data.columns[featlist]))
    feature_csv.index.name='Feature #'
    feature_csv.columns=['Feature Name']

    print(len(featlist))
    
    data_cut= data[data.columns[featlist]]
     
    data_cut.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/data-cut-to-feature-set.csv', index_label='PATIENT')
    feature_csv.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/intersecting-features-index.csv')
       
    return

def InnerFeats():
    a=input('Click and drag ENTIRE DATASET file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag SINGLE FOLD INNER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: inner_cv= pickle.load(f)
    
    folds= len(inner_cv['train'])
    thisfold=input('Which # fold is this? ')
    feats=[[0]]*folds
    
    #This is correct because we are mimicking the entire L(.) procedure as if D-1 were D.
    for i in range(folds):
        subjects=pd.DataFrame(index=inner_cv['train'][i])
        X= subjects.join(data)
        y= subjects.join(labels)

        #llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        llic= SelectFromModel(LassoLarsIC())
        llic.fit(X,y)
        feats[i]=llic.get_support(indices=True)
    
    featlist= list(set.intersection(*map(set,feats)))
    featlist.sort()
    feature_csv= pd.DataFrame(index=featlist, data= list(data.columns[featlist]))
    feature_csv.index.name='Feature #'
    feature_csv.columns=['Feature Name']

    print(len(featlist))
    
    data_cut= data[data.columns[featlist]]
        
    data_cut.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/data-cut-to-feature-set-for-inner-fold-'+str(thisfold)+'.csv', index_label='PATIENT')
    feature_csv.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/intersecting-features-index-for-inner-fold-'+str(thisfold)+'.csv')
        
    return    

def HoldoutCut():
    a=input('Click and drag FEATURE SELECTED ENTIRE DATASET file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 

    b=input('Click and drag HOLDOUT DATA file here: ')
    b=b.strip('\' ')
    hdata=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    data_cut=hdata[data.columns]
    
    data_cut.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/holdout-data-cut-to-feature-set.csv', index_label='PATIENT')
    
'''
In case we need to re-integrate individual folds and run through them five at a time:

    foldgroup=[]
    for i in range(0, len(feats), 5):
        foldgroup.append(feats[i:i+5])
    
    for i in range(len(foldgroup)):
        featlist= list(set.intersection(*map(set,foldgroup[i])))
        featlist.sort()
        
        data_cut= data[data.columns[featlist]]
'''

