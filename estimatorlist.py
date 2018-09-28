#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, neural_network


def AllEstimators(data):   
    
    #strategy: one run on defaults, 1 run on settings that worked previously, 3 runs varying # features from previously successful run
    
    nfeatsmax= len(data.columns)
    nfeatsneural= round((nfeatsmax*2/3))
    
    rf= ensemble.RandomForestClassifier()
    rf2= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    rf3= ensemble.RandomForestClassifier(max_features=10, max_depth=5, bootstrap=False)
    
    et= ensemble.ExtraTreesClassifier()
    et2= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    et3= ensemble.ExtraTreesClassifier(max_features=10, max_depth=5, bootstrap=False)
    
    kn= neighbors.KNeighborsClassifier()
    kn2= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    kn3= neighbors.KNeighborsClassifier(n_neighbors=10, p=1)

    nb= naive_bayes.GaussianNB()
    nb2= naive_bayes.GaussianNB(var_smoothing=1e-8)
    nb3= naive_bayes.GaussianNB(var_smoothing=1e-10)
    
    dt= tree.DecisionTreeClassifier()
    dt2= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    dt3= tree.DecisionTreeClassifier(max_features=10, max_depth=5, criterion='entropy')
    
    la= linear_model.Lasso()
    la2= linear_model.Lasso(alpha=0.5)
    la3= linear_model.Lasso(alpha=2.0)

    ls= svm.LinearSVC()
    ls2= svm.LinearSVC(penalty='l1', dual=False)
    ls3= svm.LinearSVC(penalty='l1', dual=False, C=2.0)

    gb= ensemble.GradientBoostingClassifier()
    gb2= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    gb3= ensemble.GradientBoostingClassifier(loss='exponential')

    nn= neural_network.MLPClassifier()
    nn2= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    nn3= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural))

    ab= ensemble.AdaBoostClassifier()
    ab2= ensemble.AdaBoostClassifier(base_estimator= ls2)
    ab3= ensemble.AdaBoostClassifier(base_estimator= rf2)

    bc= ensemble.BaggingClassifier()
    bc2= ensemble.BaggingClassifier(base_estimator=ls2)
    bc3= ensemble.BaggingClassifier(base_estimator=rf2)
    
    vc= ensemble.VotingClassifier(estimators=[('ab2', ab2),('gb2', gb2),('bc2', bc2)], voting='soft')
    vc2=ensemble.VotingClassifier(estimators=[('ab2', ab2),('gb2', gb2),('bc2', bc2)], voting='hard')
    vc3=ensemble.VotingClassifier(estimators=[('rf2', rf2),('ls2', ls2),('la2', la2)], voting='soft')
    
    estimators= {'randomforest': rf, 'randomforest2': rf2,'randomforest3': rf3,
                 'extratrees': et, 'extratrees2': et2, 'extratrees3': et3, 
                 'kneighbors': kn, 'kneighbors2': kn2, 'kneighbors3': kn3, 
                 'naivebayes': nb, 'naivebayes2': nb2, 'naivebayes3': nb3, 
                 'decisiontree': dt, 'decisiontree2': dt2, 'decisiontree3': dt3, 
                 'linearsvc': ls, 'linearsvc2': ls2, 'linearsvc3': ls3, 
                 'gboost': gb, 'gboost2': gb2, 'gboost3': gb3, 
                 'neuralnet': nn, 'neuralnet2': nn2, 'neuralnet3': nn3,
                 'adaboost': ab, 'adaboost2': ab2, 'adaboost3': ab3,
                 'bagging': bc, 'bagging2': bc2, 'bagging3': bc3,
                 'voting': vc, 'voting2': vc2, 'voting3': vc3
                 }  
   
    return
    
