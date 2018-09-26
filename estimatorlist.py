#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, neural_network

def AllEstimators(data):   
    
    #strategy: one run on defaults, 1 run on settings that worked previously, 3 runs varying # features from previously successful run
    
    nfeatsmax= len(data.columns)
    
    rf= ensemble.RandomForestClassifier()
    rf2= ensemble.RandomForestClassifier(max_features=nfeatsmax, max_depth=5,bootstrap=False)
    rf3= ensemble.RandomForestClassifier(max_features=10, max_depth=5, bootstrap=False)
    rf4= ensemble.RandomForestClassifier(max_features=5, max_depth=5, bootstrap=False)
    rf5= ensemble.RandomForestClassifier(max_features=3, max_depth=3, bootstrap=False)
    
    et= ensemble.ExtraTreesClassifier()
    et2= ensemble.ExtraTreesClassifier(max_features=nfeatsmax, max_depth=5, bootstrap=False)
    et3= ensemble.ExtraTreesClassifier(max_features=10, max_depth=5 bootstrap=False)
    et4= ensemble.ExtraTreesClassifier(max_features=5, max_depth=5 bootstrap=False)
    et5= ensemble.ExtraTreesClassifier(max_features=3, max_depth=3 bootstrap=False)
    
    kn= neighbors.KNeighborsClassifier()
    kn2= neighbors.KNeighborsClassifier(n_neighbors=nfeatsmax, p=1)
    kn3= neighbors.KNeighborsClassifier(n_neighbors=10, p=1)
    kn4= neighbors.KNeighborsClassifier(n_neighbors=5, p=1)
    kn5= neighbors.KNeighborsClassifier(n_neighbors=3, p=1)
    
    #***START HERE***
    nb= naive_bayes.GaussianNB()
    nb2= naive_bayes.GaussianNB(normalize=True, fit_intercept= False)
    nb3= naive_bayes.GaussianNB()
    nb4= naive_bayes.GaussianNB(normalize=True, fit_intercept= False)
    nb5= naive_bayes.GaussianNB()
    
    dt= tree.DecisionTreeClassifier(max_features=nfeatsmax, max_depth=5, criterion='entropy')
    dt2= tree.DecisionTreeClassifier()
    dt3= tree.DecisionTreeClassifier(max_features=10, max_depth=5, criterion='entropy')
    dt4= tree.DecisionTreeClassifier()
    dt5= tree.DecisionTreeClassifier(max_features=10, max_depth=5, criterion='entropy')
    
    la= linear_model.Lasso()
    la2= linear_model.Lasso()
    la3= linear_model.Lasso()
    la4= linear_model.Lasso()
    la5= linear_model.Lasso()

    ls= svm.LinearSVC(penalty='l1', dual=False)
    ls2= svm.LinearSVC()
    ls3= svm.LinearSVC(penalty='l1', dual=False, C=2.0)
    ls4= svm.LinearSVC()
    ls5= svm.LinearSVC(penalty='l1', dual=False, C=2.0)

    gb= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    gb2= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    gb3= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    gb4= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)
    gb5= ensemble.GradientBoostingClassifier(loss='exponential', max_depth=2)

    nn= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    nn2= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    nn3= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    nn4= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)
    nn5= neural_network.MLPClassifier(hidden_layer_sizes=(nfeatsneural,nfeatsneural,nfeatsneural), learning_rate_init=0.0001, max_iter=500)

    ab= ensemble.AdaBoostClassifier()
    ab2= ensemble.AdaBoostClassifier()
    ab3= ensemble.AdaBoostClassifier()
    ab4= ensemble.AdaBoostClassifier()
    ab5= ensemble.AdaBoostClassifier()

    bc= ensemble.BaggingClassifier(base_estimator=rf)
    bc2= ensemble.BaggingClassifier(base_estimator=rf2)
    bc3= ensemble.BaggingClassifier(base_estimator=ls)
    bc4= ensemble.BaggingClassifier(base_estimator=rf2)
    bc5= ensemble.BaggingClassifier(base_estimator=ls)
    
    vc= ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='soft')
    vc2=ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='hard')
    vc3=ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='hard')
    vc4=ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='hard')
    vc5=ensemble.VotingClassifier(estimators=[('ab', ab),('gb', gb),('bc', bc)], voting='hard')
    
    estimators= {'randomforest': rf, 'randomforest2': rf2,'randomforest3': rf3, 'randomforest4': rf4, 'randomforest5': rf5,
                 'extratrees': et, 'extratrees2': et2, 'extratrees3': et3, 'extratrees4': et4, 'extratrees5': et5,
                 'kneighbors': kn, 'kneighbors2': kn2, 'kneighbors3': kn3, 'kneighbors4': kn4, 'kneighbors5': kn5,
                 'naivebayes': nb, 'naivebayes2': nb2, 'naivebayes3': nb3, 'naivebayes4': nb4, 'naivebayes5': nb5, 
                 'decisiontree': dt, 'decisiontree2': dt2, 'decisiontree3': dt3, 'decisiontree4': dt4, 'decisiontree5': dt5,
                 'linearsvc': ls, 'linearsvc2': ls2, 'linearsvc3': ls3, 'linearsvc4': ls4, 'linearsvc5': ls5,  
                 'gboost': gb, 'gboost2': gb2, 'gboost3': gb3, 'gboost4': gb4, 'gboost5': gb5,
                 'neuralnet': nn, 'neuralnet2': nn2, 'neuralnet3': nn3, 'neuralnet4': nn4, 'neuralnet5': nn5, 
                 'adaboost': ab, 'adaboost2': ab2, 'adaboost3': ab3, 'adaboost4': ab4, 'adaboost5': ab5, 
                 'bagging': bc, 'bagging2': bc2, 'bagging3': bc3, 'bagging4': bc4, 'bagging5': bc5,
                 'voting': vc, 'voting2': vc2, 'voting3': vc3, 'voting4': vc4, 'voting5': vc5
                 }  
   
    return
    
def Bestimator():
    
