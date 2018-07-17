#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, neural_network
 
 
def Bestimator():
    rf= ensemble.RandomForestClassifier(max_features=10, max_depth=5, n_jobs=3, bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=10, max_depth=5, n_jobs=3, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=3, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=10, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(7,7,7), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf, n_jobs=3)
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    
    estimators= {'randomforest': rf,
                 'extratrees': et,
                 'kneighbors': kn,
                 'naivebayes': nb,
                 'decisiontree': dt,
                 'linearsvc': ls,
                 'gboost': gb,
                 'neuralnet': nn,
                 'adaboost': ab,
                 'voting': vc,
                 'bagging': bc}
                 
    a=input('Click and drag POST FEATURE SELECTION DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag OUTER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: ocv= pickle.load(f)

    results= {'estimator':[], 
              'subjects':[], 
              'labels':[], 
              'predictions':[], 
              'scores':[], 
              'attempts':[]}
    
    hresults= {'estimator':[], 
               'subjects':[], 
               'labels':[], 
               'predictions':[], 
               'scores':[], 
               'attempts':[]}
           
    for j,k in zip(estimators.keys(), estimators.values()):           
        X,y= data,labels['HAMD 1=REMIT']
        k.fit(X, y)
        predict= k.predict(X)
        scores= [1 if a==b else 0 for a,b in zip(predict, y)]          
        results['estimator'].extend([j]*len(X))
        results['subjects'].extend(labels.index)
        results['labels'].extend(y)
        results['predictions'].extend(predict)
        results['scores'].extend(scores)
        results['attempts'].extend([1]*len(X))
        
        hX,hy= hdata, hlabels['HAMD-17 Remit']
        hpredict= k.predict(hX)
        hscores= [1 if a==b else 0 for a,b in zip(hpredict, hy)]        
        hresults['estimator'].extend([j]*len(hX))
        hresults['subjects'].extend(hlabels.index)
        hresults['labels'].extend(hy)
        hresults['predictions'].extend(hpredict)
        hresults['scores'].extend(hscores)
        hresults['attempts'].extend([1]*len(hX))

    df=pd.DataFrame.from_dict(results).set_index('subjects')
    hdf=pd.DataFrame.from_dict(hresults).set_index('subjects')

    df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/all-data-results.csv')
    hdf.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/holdout-results.csv')

    estimator= df.groupby('estimator').sum()
    accuracy= (estimator['scores']/estimator['attempts'])*100
    print(accuracy)
    pmax= accuracy.idxmax(axis=1)
    print('\nAll data final run accuracy: {}\n'.format(pmax))
    
    hestimator= hdf.groupby('estimator').sum()
    haccuracy= (hestimator['scores']/hestimator['attempts'])*100
    print(haccuracy)
    

def BestimatorOld():
    rf= ensemble.RandomForestClassifier(max_features=10, max_depth=5, n_jobs=3, bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=10, max_depth=5, n_jobs=3, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=3, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=10, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(7,7,7), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf, n_jobs=3)
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    
    estimators= {'randomforest': rf,
                 'extratrees': et,
                 'kneighbors': kn,
                 'naivebayes': nb,
                 'decisiontree': dt,
                 'linearsvc': ls,
                 'gboost': gb,
                 'neuralnet': nn,
                 'adaboost': ab,
                 'voting': vc,
                 'bagging': bc
                 }
    
    datafile= input('Click and drag DATA file: ')
    datafile= datafile.strip('\' ')
    data= pd.read_csv(datafile, encoding='utf-8').set_index('PATIENT')  
    
    labelfile= input('Click and drag LABELS file: ')
    labelfile= labelfile.strip('\' ')
    labels= pd.read_csv(labelfile, encoding='utf-8').set_index('PATIENT')  
   
    hdatafile= input('Click and drag HOLDOUT DATA file: ')
    hdatafile= hdatafile.strip('\' ')
    hdata= pd.read_csv(hdatafile, encoding='utf-8').set_index('PATIENT') 
   
    hlabelfile= input('Click and drag HOLDOUT LABELS file: ')
    hlabelfile= hlabelfile.strip('\' ')
    hlabels= pd.read_csv(hlabelfile, encoding='utf-8').set_index('PATIENT')  
   
    results= {'estimator':[], 
              'subjects':[], 
              'labels':[], 
              'predictions':[], 
              'scores':[], 
              'attempts':[]}
    
    hresults= {'estimator':[], 
               'subjects':[], 
               'labels':[], 
               'predictions':[], 
               'scores':[], 
               'attempts':[]}
           
    for j,k in zip(estimators.keys(), estimators.values()):           
        X,y= data,labels['HAMD 1=REMIT']
        k.fit(X, y)
        predict= k.predict(X)
        scores= [1 if a==b else 0 for a,b in zip(predict, y)]          
        results['estimator'].extend([j]*len(X))
        results['subjects'].extend(labels.index)
        results['labels'].extend(y)
        results['predictions'].extend(predict)
        results['scores'].extend(scores)
        results['attempts'].extend([1]*len(X))
        
        hX,hy= hdata, hlabels['HAMD-17 Remit']
        hpredict= k.predict(hX)
        hscores= [1 if a==b else 0 for a,b in zip(hpredict, hy)]        
        hresults['estimator'].extend([j]*len(hX))
        hresults['subjects'].extend(hlabels.index)
        hresults['labels'].extend(hy)
        hresults['predictions'].extend(hpredict)
        hresults['scores'].extend(hscores)
        hresults['attempts'].extend([1]*len(hX))

    df=pd.DataFrame.from_dict(results).set_index('subjects')
    hdf=pd.DataFrame.from_dict(hresults).set_index('subjects')

    df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/all-data-results.csv')
    hdf.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/holdout-results.csv')

    estimator= df.groupby('estimator').sum()
    accuracy= (estimator['scores']/estimator['attempts'])*100
    print(accuracy)
    pmax= accuracy.idxmax(axis=1)
    print('\nAll data final run accuracy: {}\n'.format(pmax))
    
    hestimator= hdf.groupby('estimator').sum()
    haccuracy= (hestimator['scores']/hestimator['attempts'])*100
    print(haccuracy)
    
    return
