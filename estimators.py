#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, neural_network

def EntireDataset():
    a=input('Click and drag FEATURE SELECTED ENTIRE DATASET file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels_df=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    labels=np.array(labels_df[labels_df.columns[0]])
       
    nfeatsmax= len(data.columns)
    nfeatsneural= round((nfeatsmax*2/3))
    
    rf= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural,), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf)
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    
    estimators= {#'randomforest': rf,
                 #'extratrees': et,
                 #'kneighbors': kn,
                 #'naivebayes': nb,
                 #'decisiontree': dt,
                 'linearsvc': ls,
                 #'gboost': gb,
                 #'neuralnet': nn,
                 #'adaboost': ab,
                 #'bagging': bc,
                 #'voting': vc,
                 }   
    
    results= {'estimator':[], 
              'subjects':[], 
              'labels':[], 
              'predictions':[], 
              'scores':[], 
              'attempts':[]}

    for j,k in zip(estimators.keys(), estimators.values()):
        k.fit(data, labels) 
        predict_train= k.predict(data)
        train_scores= [1 if x==y else 0 for x,y in zip(labels, predict_train)]            
        results['estimator'].extend([j]*len(data))
        results['subjects'].extend(data.index)
        results['labels'].extend(labels)
        results['predictions'].extend(predict_train)
        results['scores'].extend(train_scores)
        results['attempts'].extend([1]*len(data))

    results_df=pd.DataFrame.from_dict(results).set_index('subjects')    
    results_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/entire_dataset_results.csv')
    
    with open('/media/james/ext4data/current/projects/pfizer/combined-study/trainedclassifier.pickle', 'wb') as f: pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)
     
    print('ENTIRE DATASET ACCURACY')
    trd= results_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)    
    
    return

def HoldoutDataset():
    a=input('Click and drag HOLDOUT FEATURE SELECTED DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag HOLDOUT LABELS file here: ')
    b=b.strip('\' ')
    labels_df=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    labels=np.array(labels_df[labels_df.columns[0]])
    
    a=input('Click and drag TRAINED CLASSIFIER file here: ')
    a=a.strip('\' ')
    with open(a, 'rb') as f: k= pickle.load(f)
        
    results= {'estimator':[], 
              'subjects':[], 
              'labels':[], 
              'predictions':[], 
              'scores':[], 
              'attempts':[]}

    j=input('Type NAME of classifier: ')
    predict_train= k.predict(data)
    train_scores= [1 if x==y else 0 for x,y in zip(labels, predict_train)]            
    results['estimator'].extend([j]*len(data))
    results['subjects'].extend(data.index)
    results['labels'].extend(labels)
    results['predictions'].extend(predict_train)
    results['scores'].extend(train_scores)
    results['attempts'].extend([1]*len(data))

    results_df=pd.DataFrame.from_dict(results).set_index('subjects')    
    results_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/holdout_dataset_results.csv')
        
    print('HOLDOUT DATASET ACCURACY')
    trd= results_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)    
    
    return

def OuterFolds():
    a=input('Click and drag FEATURE SELECTED ENTIRE DATASET file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag OUTER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: outer_cv= pickle.load(f)
   
    folds= len(outer_cv['train'])    
    nfeatsmax= len(data.columns)
    nfeatsneural= round((nfeatsmax*2/3))
    
    rf= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf)
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    
    #We can just copypasta these with slightly different hyperparameters to make a nice set of tests that can all be run at once with a nice flat structure.
    vc2=ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='hard')
    
    estimators= {'randomforest': rf,
                 'extratrees': et,
                 'kneighbors': kn,
                 'naivebayes': nb,
                 'decisiontree': dt,
                 'linearsvc': ls,
                 'gboost': gb,
                 'neuralnet': nn,
                 'adaboost': ab,
                 'bagging': bc,
                 'voting: softball': vc,
                 'voting 2: hardball': vc2,
                 }  
   
    train_results= {'fold':[], 'estimator':[], 'subjects':[], 
                    'labels':[], 'predictions':[], 'scores':[], 
                    'attempts':[]}
                    
    test_results= {'fold':[], 'estimator':[], 'subjects':[], 
                   'labels':[], 'predictions':[], 'scores':[], 
                   'attempts':[]}
    
    for i in range(folds):
        train_ids=pd.DataFrame(index=outer_cv['train'][i])
        X_train= train_ids.join(data)
        y_train_df= train_ids.join(labels)
        y_train= np.array(y_train_df[y_train_df.columns[0]])
        
        test_ids=pd.DataFrame(index=outer_cv['test'][i])
        X_test= test_ids.join(data)
        y_test_df= test_ids.join(labels)
        y_test= np.array(y_test_df[y_test_df.columns[0]])

        for j,k in zip(estimators.keys(), estimators.values()):
            k.fit(X_train, y_train) 
                      
            predict_train= k.predict(X_train)
            train_scores= [1 if x==y else 0 for x,y in zip(y_train, predict_train)]            
            train_results['fold'].extend([i+1]*len(X_train))
            train_results['estimator'].extend([j]*len(X_train))
            train_results['subjects'].extend(train_ids.index)
            train_results['labels'].extend(y_train)
            train_results['predictions'].extend(predict_train)
            train_results['scores'].extend(train_scores)
            train_results['attempts'].extend([1]*len(X_train))

            predict_test= k.predict(X_test)
            test_scores= [1 if x==y else 0 for x,y in zip(y_test, predict_test)]         
            test_results['fold'].extend([i+1]*len(X_test))
            test_results['estimator'].extend([j]*len(X_test))
            test_results['subjects'].extend(test_ids.index)
            test_results['labels'].extend(y_test)
            test_results['predictions'].extend(predict_test)
            test_results['scores'].extend(test_scores)
            test_results['attempts'].extend([1]*len(X_test))

    train_df=pd.DataFrame.from_dict(train_results).set_index('subjects')
    test_df=pd.DataFrame.from_dict(test_results).set_index('subjects')
    
    train_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/outer_train_results.csv')
    test_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/outer_test_results.csv')
    
    print('TRAIN RESULT')
    trd= train_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)
    trmax= trsum.idxmax(axis=1)
    print('\nBest train: {}\n'.format(trmax))

    print('TEST RESULT')
    ted= test_df.groupby('estimator').sum()
    tesum= (ted['scores']/ted['attempts'])*100
    print(tesum)
    temax= tesum.idxmax(axis=1)
    print('\nBest test: {}\n'.format(temax))
    
    return


def InnerFolds():
    a=input('Click and drag FEATURE SELECTED SINGLE FOLD DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag SINGLE FOLD INNER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: outer_cv= pickle.load(f)
   
    thisfold=input('Which fold is this? ')
    folds= len(outer_cv['train'])    
    nfeatsmax= len(data.columns)
    nfeatsneural= round((nfeatsmax*2/3))
    
    rf= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf)
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
                 'bagging': bc,
                 'voting': vc
                 }  
   
    train_results= {'fold':[], 'estimator':[], 'subjects':[], 
                    'labels':[], 'predictions':[], 'scores':[], 
                    'attempts':[]}
                    
    test_results= {'fold':[], 'estimator':[], 'subjects':[], 
                   'labels':[], 'predictions':[], 'scores':[], 
                   'attempts':[]}
    
    for i in range(folds):
        train_ids=pd.DataFrame(index=outer_cv['train'][i])
        X_train= train_ids.join(data)
        y_train_df= train_ids.join(labels)
        y_train= np.array(y_train_df[y_train_df.columns[0]])
        
        test_ids=pd.DataFrame(index=outer_cv['test'][i])
        X_test= test_ids.join(data)
        y_test_df= test_ids.join(labels)
        y_test= np.array(y_test_df[y_test_df.columns[0]])

        for j,k in zip(estimators.keys(), estimators.values()):
            k.fit(X_train, y_train)
                                  
            predict_train= k.predict(X_train)
            train_scores= [1 if x==y else 0 for x,y in zip(y_train, predict_train)]            
            train_results['fold'].extend([i+1]*len(X_train))
            train_results['estimator'].extend([j]*len(X_train))
            train_results['subjects'].extend(train_ids.index)
            train_results['labels'].extend(y_train)
            train_results['predictions'].extend(predict_train)
            train_results['scores'].extend(train_scores)
            train_results['attempts'].extend([1]*len(X_train))

            predict_test= k.predict(X_test)
            test_scores= [1 if x==y else 0 for x,y in zip(y_test, predict_test)]         
            test_results['fold'].extend([i+1]*len(X_test))
            test_results['estimator'].extend([j]*len(X_test))
            test_results['subjects'].extend(test_ids.index)
            test_results['labels'].extend(y_test)
            test_results['predictions'].extend(predict_test)
            test_results['scores'].extend(test_scores)
            test_results['attempts'].extend([1]*len(X_test))        
            
    train_df=pd.DataFrame.from_dict(train_results).set_index('subjects')
    test_df=pd.DataFrame.from_dict(test_results).set_index('subjects')
    
    train_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/inner_train_results_fold_'+str(thisfold)+'.csv')
    test_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/inner_test_results_fold_'+str(thisfold)+'.csv')
    
    
    print('TRAIN RESULT')
    trd= train_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)
    trmax= trsum.idxmax(axis=1)
    print('\nBest train: {}\n'.format(trmax))

    print('TEST RESULT')
    ted= test_df.groupby('estimator').sum()
    tesum= (ted['scores']/ted['attempts'])*100
    print(tesum)
    temax= tesum.idxmax(axis=1)
    print('\nBest test: {}\n'.format(temax))
    
    return

def InnerHoldout():     
    a=input('Click and drag FEATURE SELECTED SINGLE FOLD DATA file here: ')
    a=a.strip('\' ')
    data=pd.read_csv(a, encoding='utf-8').set_index('PATIENT') 
    
    b=input('Click and drag LABELS file here: ')
    b=b.strip('\' ')
    labels=pd.read_csv(b, encoding='utf-8').set_index('PATIENT') 
    
    c=input('Click and drag OUTER CV file here: ')
    c=c.strip('\' ')
    with open(c, 'rb') as f: outer_cv= pickle.load(f)
   
    thisfold= int(input('Which fold is this? '))
    
    nfeatsmax= len(data.columns)
    nfeatsneural= round((nfeatsmax*2/3))
    
    rf= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    et= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    kn= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    nb= naive_bayes.GaussianNB()
    dt= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    ls= svm.LinearSVC(penalty='l1', dual=False)
    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    ab= ensemble.AdaBoostClassifier()
    bc= ensemble.BaggingClassifier(base_estimator=rf)
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    
    estimators= {#'randomforest': rf,
                 #'extratrees': et,
                 #'kneighbors': kn,
                 #'naivebayes': nb,
                 #'decisiontree': dt,
                 'linearsvc': ls,
                 #'gboost': gb,
                 #'neuralnet': nn,
                 #'adaboost': ab,
                 #'bagging': bc,
                 #'voting': vc
                 }  
   
    train_results= {'fold':[], 'estimator':[], 'subjects':[], 
                    'labels':[], 'predictions':[], 'scores':[], 
                    'attempts':[]}
                    
    test_results= {'fold':[], 'estimator':[], 'subjects':[], 
                   'labels':[], 'predictions':[], 'scores':[], 
                   'attempts':[]}
    
    train_ids=pd.DataFrame(index=outer_cv['train'][thisfold-1])
    X_train= train_ids.join(data)
    y_train_df= train_ids.join(labels)
    y_train= np.array(y_train_df[y_train_df.columns[0]])
        
    test_ids=pd.DataFrame(index=outer_cv['test'][thisfold-1])
    X_test= test_ids.join(data)
    y_test_df= test_ids.join(labels)
    y_test= np.array(y_test_df[y_test_df.columns[0]])

    for j,k in zip(estimators.keys(), estimators.values()):
        k.fit(X_train, y_train) 
                      
        predict_train= k.predict(X_train)
        train_scores= [1 if x==y else 0 for x,y in zip(y_train, predict_train)]            
        train_results['fold'].extend([thisfold]*len(X_train))
        train_results['estimator'].extend([j]*len(X_train))
        train_results['subjects'].extend(train_ids.index)
        train_results['labels'].extend(y_train)
        train_results['predictions'].extend(predict_train)
        train_results['scores'].extend(train_scores)
        train_results['attempts'].extend([1]*len(X_train))

        predict_test= k.predict(X_test)
        test_scores= [1 if x==y else 0 for x,y in zip(y_test, predict_test)]         
        test_results['fold'].extend([thisfold]*len(X_test))
        test_results['estimator'].extend([j]*len(X_test))
        test_results['subjects'].extend(test_ids.index)
        test_results['labels'].extend(y_test)
        test_results['predictions'].extend(predict_test)
        test_results['scores'].extend(test_scores)
        test_results['attempts'].extend([1]*len(X_test))

    train_df=pd.DataFrame.from_dict(train_results).set_index('subjects')
    test_df=pd.DataFrame.from_dict(test_results).set_index('subjects')
    
    train_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/inner_holdout_train_results_fold_'+str(thisfold)+'.csv')
    test_df.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/combined-study/inner_holdout_test_results_fold_'+str(thisfold)+'.csv')
    
    with open('/media/james/ext4data/current/projects/pfizer/combined-study/trainedclassifier_innerfold_'+str(thisfold)+'.pickle', 'wb') as f: pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)
    
    print('D_-j RESULT')
    trd= train_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)
    trmax= trsum.idxmax(axis=1)
    print('\nBest train: {}\n'.format(trmax))

    print('D_j (holdout for estimating model quality) RESULT')
    ted= test_df.groupby('estimator').sum()
    tesum= (ted['scores']/ted['attempts'])*100
    print(tesum)
    temax= tesum.idxmax(axis=1)
    print('\nBest test: {}\n'.format(temax))
    
    return
