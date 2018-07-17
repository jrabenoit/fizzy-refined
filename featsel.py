#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoLarsIC
import copy, pickle
import numpy as np
import pandas as pd

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.
#To flatten feature list and get frequencies: 
    #[item for sublist in icv['Feature Indices'] for item in sublist]
    #from collections import Counter
    #b=dict(Counter(a))


def InnerFeats():
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/icv.pickle','rb') as f: icv=pickle.load(f)       
    X_train= icv['X_train']
    X_test= icv['X_test']
    y_train= icv['y_train']
    y_test= icv['y_test']
    train_indices= icv['train_indices']
    test_indices= icv['test_indices']
    
    folds= len(icv['X_train'])
    feats=[[0]]*folds
    
    for i in range(folds):
        print(i+1)
        subjects=len(X_train[i])
        
        #skb= SelectKBest(k='all')
        #skb.fit(X_train[i], y_train[i])
        #X_train_feats= skb.transform(X_train[i])
        #X_test_feats= skb.transform(X_test[i])

        #RFECV option
        #rfe= RFECV(RandomForestClassifier(), step=1, n_jobs=3)
        #rfe.fit(X_train_skb, y_train[i])
        #feats[i]=rfe.get_support(indices=True)
        #X_train_feats= rfe.transform(X_train_skb)
        #X_test_feats= rfe.transform(X_test_skb)

        #LassoLarsIC option
        llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        llic.fit(X_train[i], y_train[i])
        feats[i]=llic.get_support(indices=True)
        X_train_feats= llic.transform(X_train[i])
        X_test_feats= llic.transform(X_test[i])
        
        X_train[i]= np.array(X_train_feats)
        X_test[i]= np.array(X_test_feats)
    
    featdict={'Feature Indices':feats, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'train_indices':train_indices, 'test_indices':test_indices}
        
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/icvfeats.pickle','wb') as f: pickle.dump(featdict, f, pickle.HIGHEST_PROTOCOL)
        
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
    with open(c, 'rb') as f: ocv= pickle.load(f)
    
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/ocv.pickle','rb') as f: ocv=pickle.load(f)
    
    folds= len(ocv['X_train'])
    feats=[[0]]*folds
    
    for i in range(folds):
        subjects=pd.DataFrame(index=ocv['X_train'][i])
        X_train= subjects.join(data)
        y_train= subjects.join(labels)

        llic= SelectFromModel(LassoLarsIC(criterion='bic'))
        llic.fit(X_train, y_train)
        feats[i]=llic.get_support(indices=True)
        
        #skb= SelectKBest(k='all')
        #skb.fit(X_train, y_train)
        #feats[i]=skb.get_support(indices=True)
    
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
    
def HoldoutFeats():
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/holdoutocv.pickle','rb') as f: ocv=pickle.load(f)       
    X_train= ocv['X_train']
    X_test= ocv['X_test']
    y_train= ocv['y_train']
    y_test= ocv['y_test']
    train_indices= ocv['train_indices']
    test_indices= ocv['test_indices']
    
    folds= len(ocv['X_train'])
    feats=[[0]]*folds
    
    skb= SelectKBest(k='all')
    #llic= SelectFromModel(LassoLarsIC(criterion='bic'))

    for i in range(folds):
        subjects=len(X_train[i])
               
        skb.fit(X_train[i], y_train[i])
        feats[i]=skb.get_support(indices=True)
        X_train_feats= skb.transform(X_train[i])
        X_test_feats= skb.transform(X_test[i])
        
        #llic.fit(X_train[i], y_train[i])
        #feats[i]=llic.get_support(indices=True)
        #X_train_feats= llic.transform(X_train[i])
        #X_test_feats= llic.transform(X_test[i])
        
        X_train[i]= np.array(X_train_feats)
        X_test[i]= np.array(X_test_feats)
    
    featdict={'Feature Indices':feats, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'train_indices':train_indices, 'test_indices':test_indices}
        
    with open('/media/james/ext4data1/current/projects/pfizer/combined-study/holdoutocvfeats.pickle','wb') as f: pickle.dump(featdict, f, pickle.HIGHEST_PROTOCOL)
        
    return
