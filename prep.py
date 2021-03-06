#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np
import pathlib

def Misc():    
    
    #VARIABLES
    
    #Encode a categorical variable as an integers (not onehot or dummy)
    demow['ETHNIC']=pd.Categorical(demow['ETHNIC']).codes

    #Quick way to encode/binarize
    df.set_index('PATIENT')
    df['ones']=1
    df.pivot(columns='MEDGNX', values='ones')
    df=df.fillna(value=0)

    #Recode a variable
    df['Severity of illness']=df['Severity of illness'].replace(to_replace='Borderline ill', value='Borderline mentally ill')

    #PIVOTS & JOINS

    #pivots table
    d60p= d60.pivot(index='PATIENT',columns='TESTS',values='VALN')
    
    #pivot each column of 'lab' to its own table (lab1, lab2, etc)
    lab1= lab.pivot(index='PATIENT',columns='LPARM',values='LVALN')
    
    #relabels columns so join works
    lab1.columns='lab1-'+lab1.columns
    
    #joins tables 
    labs=lab1.join([lab2, lab3])

    #Returns vals in df1 that are ALSO in df2
    np.intersect1d(df1['PATIENT'],df2['PATIENT'])

    #Returns vals in df1 that are NOT in df2
    np.setdiff1d(df1['PATIENT'],df2['PATIENT'])

    #Remove a study's participants (e.g. if missing data)
    dfs3=dfs2.drop(labels=df.index, errors='ignore')
    

    #MISC
    
    #Descriptive statistics: central tendency, dispersion and shape
    df['HAMD Total'].describe()
    
    #drops duplicates if both column 1 AND column 2 have the same row value
    lab=lab.drop_duplicates(subset=['PATIENT', 'LPARM'], keep='first')

    #Quick way to sample a class to even out classes
    a=labels.columns[0]
    b=labels[a].value_counts()
    df2=labels.loc[labels[a]==0]
    df3=labels.loc[labels[a]==1]
    df4=df2.sample(min(b))
    df5=df3.sample(min(b))
    df6=[df4, df5] 
    df7=pd.concat(df6)
    df8=df7.sort_index()

    #Quick way to cut data to labels
    data=data[data.index.isin(labels.index)].sort_index()


    #Sample for holdout set at same frequency of labels as dataset
    #Using ~ to cut the holdout set out of the main set, or not using it to get just the holdout set on its own


holdout=labels.groupby('HAM-D17 1=REMIT').apply(pd.DataFrame.sample, frac=0.1).reset_index(level='HAM-D17 1=REMIT', drop=True).sort_index()
    
holdout_labels=labels[data.index.isin(holdout.index)].sort_index()
labels_excluding_holdout=labels[~data.index.isin(holdout.index)].sort_index()
holdout_data=data[data.index.isin(holdout.index)].sort_index() 
data_excluding_holdout=data[~data.index.isin(holdout.index)].sort_index()
    
holdout_labels.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/refined-combined-study/holdout-labels.csv', index_label='PATIENT')
labels_excluding_holdout.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/refined-combined-study/labels_excluding_holdout.csv', index_label='PATIENT')
holdout_data.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/refined-combined-study/holdout-data.csv', index_label='PATIENT')
data_excluding_holdout.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/refined-combined-study/data_excluding_holdout.csv', index_label='PATIENT')
    
    #Cgi ordered categories, -1 indicates NaN
    df=df.set_index('PATIENT')
    cat=['Normal, not at all ill',
    'Borderline mentally ill',
    'Mildly ill',  
    'Moderately ill',
    'Markedly ill',
    'Severely ill',
    'Among the most extremely ill']
    df['CGI SEVERITY']=pd.Categorical(df['Severity of illness'], categories=cat).codes       
    
    df[df['CGI SEVERITY']==-1]
    >>>output: 89CHQS
    df=df.drop(['89CHQS'])
    
    #CGI process
    info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/combined-study/3151a1-3364-csv/deid_cgi.csv', encoding='utf-8') 

    info=info[info['CPENM']=='BASELINE DAY -1'] 

    info=info.drop_duplicates(subset='PATIENT', keep='first')   

    data= info.pivot(index='PATIENT',columns='TESTS',values='VALX') 

    data.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/3151a1-3364-csv/deid_cgi_ready.csv',index_label='PATIENT')  
    
    #demow ordered sex/race
    cat=['Male', 'Female']
    
    #split off a variable
    dftherdur=df['THERDUR'].to_frame()
    del df['THERDUR']
    
    #ethnicity ordering- alphabetical like NIH
    eth=['American Indian or Alaska Native',
     'Asian',  
     'Black or African American',
     'Hispanic or Latino',
     'Middle Eastern or North African',
     'Native Hawaiian or Other Pacific Islander',
     'Other',
     'White']

    #sex ordering- alphabetical
    sex=['Female','Male']

    #ethnic recode
    df1['ETHNIC RECODE']=df1['ETHNIC RECODE'].replace({
 'American Indian or Alaska Native':'American Indian or Alaska Native',
 'Arabic':'Middle Eastern or North African',
 'Asian':'Asian',
 'Black':'Black or African American',
 'Black or African American':'Black or African American',
 'Chinese':'Asian',
 'Hispanic':'Hispanic or Latino',
 'Indian':'Asian',
 'Korean':'Asian',
 'Native American':'American Indian or Alaska Native',
 'Native Hawaiian or Other Pacific Islander':'Native Hawaiian or Other Pacific Islander',
 'Oriental(Asian)':'Asian',
 'Other':'Other',
 'Other: (Mixed race)':'Other',
 'Other: Alaskan Native':'American Indian or Alaska Native',
 'Other: Coloured.':'Black or African American',
 'Other: Mid eastern':'Middle Eastern or North African',
 'Other: Middle eastern':'Middle Eastern or North African',
 'Other: Mixed':'Other',
 'Other: Mixed race.':'Other',
 'Other: Mixed.':'Other',
 'Other: Panaminian.':'Hispanic or Latino',
 'Other: Russian':'White',
 'Other: XXXXXXXXXX':'Other',
 'Other:Bi-Racial.':'Other',
 'Other:Brazilian':'Hispanic or Latino',
 'Other:Cauc/Asian-pacific islander':'Other',
 'Other:INDIAN':'Asian',
 'Other:Mixed':'Other',
 'Other:Mixed race.':'Other',
 'Other:Mixed.':'Other',
 'Other:Slavic':'White',
 'Taiwanese':'Asian',
 'White':'White'})
    
    return

def Labeler():
    hamd=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/303-data/deid_hamd17a.csv')
    df['HAMD 1=REMIT']=np.where(df['HAMD-17 questions Total score derived']<=7, 1, 0)
            
    df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/class-labels.csv', index_label='PATIENT')
        
    return


def GroupDefiner():
    labels=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-d60-placebo-remitters.csv', encoding='utf-8').set_index('PATIENT').sort_index()
    placebos=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/placebo-patients.csv').set_index('PATIENT').sort_index()
    therapy= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/therapy-60-completed.csv').set_index('PATIENT').sort_index()
    
    placebos=placebos[placebos['TPNAME']=='Placebo']
    
    therapy=therapy[therapy['THERDUR>=60']==1]   
    
    final= labels.join([placebos, therapy], how='inner')
    
    del final['TPNAME']
    del final['THERDUR>=60']
    
    final.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/labels-final.csv', index_label='PATIENT')

    return
    
    
def Homeopathy():
    #Cuts all tables to subjects in labels-final
    patients=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-final.csv', encoding='utf-8').set_index('PATIENT').index
    
    path= '/media/james/ext4data1/current/projects/pfizer/303-data-baseline/'
    csvs= os.listdir(path)
    for i in csvs:
        a= pd.read_csv(path+i)
        b= a[a['PATIENT'].isin(patients)]
        b= b.set_index('PATIENT')
        b.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/303-data-baseline/cut-'+str(i), index_label='PATIENT')
    
    return

#>>>
#NOW /CUT/ENCODE THE TABLES DOWN MANUALLY
#>>>

def Binarizer():
    #Use if you're making binarized variables
    csv= ['deid_adverse', 'deid_aemeddra', 'deid_medhist', 'deid_medhist2', 'deid_nsmed', 'deid_othtrt']
    for i in csv:
        info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/3151A1-303-csv/'+str(i)+'.csv', encoding='utf-8')
        a= info.set_index(['PATIENT'])
        b= pd.get_dummies(a)        
        d= {}
        for j in list(set(b.index)):
            d[j]= b.loc[j].values.flatten()
        
        maxlen=len(d[max(d, key=lambda k: len(d[k]))])            
        for m in d:
            d[m]=np.append(d[m], [0]*(maxlen-len(d[m])))        
        d= pd.DataFrame.from_dict(d, orient='index')
        d.columns=list(b.columns)*scipy.stats.mode(b.index).count[0]       
        d.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/vecs/vecs_'+str(i)+'.csv', index_label='PATIENT')
        #this gives a dataframe with all variables binarized 

    return


def Harvester():
    '''Because it's a combine. Aha. Ha.'''
    #But seriously, joins all tables together by patient row
    
    info= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-d60-placebo-remitters.csv', encoding='utf-8').set_index('PATIENT').drop('GROUPLABEL', axis=1)
    
    path= '/media/james/ext4data1/current/projects/pfizer/303-data-baseline-final/'
    csvs= os.listdir(path)    
    for i in csvs:
        a=pd.read_csv(path+i).set_index('PATIENT')
        info=info.join(a, how='inner')
    
    info.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/joined-vecs.csv', index_label='PATIENT')
    
    return

def CombineStudies():
    '''Combines all studies in a directory with the same column headers'''
    
    basedir=input('Click and drag DIRECTORY here: ')
    root=basedir.strip('\' ')
    dirname= os.path.basename(root)
    
    basefiles=[]
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            fpath= os.path.join(path, name)
            basefiles=basefiles+[fpath]
    
    combinedframe= pd.DataFrame()
    
    for i in basefiles:
        print(i)
        data=pd.read_csv(i, encoding='utf-8').set_index('PATIENT')
        combinedframe=pd.concat([combinedframe,data])
    
    combinedframe.to_csv(path_or_buf='/media/james/ext4data/current/projects/pfizer/refined-combined-study/Data/'+dirname+'.csv')
        

