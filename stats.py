#!/usr/bin/env python3 

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pprint, itertools, pickle, random, statistics
from sklearn import metrics

def Roc():
    a=input('Click and drag desired TEST RESULTS file (usually entire_dataset_results or holdout_test_results: ')
    a=a.strip('\' ')
    results=pd.read_csv(a).set_index('subjects')
    
    labels= results['labels']
    predictions=results['predictions']
    
    fpr, tpr, thresholds= metrics.roc_curve(labels, predictions, pos_label=1)
    
    print('fpr: {} \ntpr: {}\n'.format(fpr[1]*100, tpr[1]*100))
    
    auc = "%.2f" % metrics.auc(fpr, tpr)
    title = 'ROC Curve, AUC = '+str(auc)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, 'darkorange', label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()
        
    return
    
def HardPlace():

    return
