import pandas as pd
import numpy as np
import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from collections import Counter
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.manifold
import time
from sklearn.model_selection import train_test_split
import base64
import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import warnings
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

###########################
#ORDER OF DATA LOADING FOR REPRODUCING RESULT

#---------Load data as per the order described below .
#---------for each file load rumour first and then non-rumour

#--------Training
#germanwing
#ottawa
#sydney
#charlie
#----------Testing= Ferguson

#--------Training
#germanwing
#ottawa
#sydney
#ferguson
#----------Testing= charlie

#--------Training
#charlie
#ottawa
#sydney
#fergu
#----------Testing= Germanwings

#--------Training
#charlie
#german
#sydney
#fergu
#----------Testing= Ottawa

#--------Training
#charlie
#ferman
#ottawa
#fergu
#----------Testing= sydney

#######################################
#Data Loading
print("\n Loading Data...")
#rumour = Label 0 , Non-Rumour =Label 1
d1=pd.read_csv('g_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0

d2=pd.read_csv('g_non_rumour.csv',header=0,encoding = "ISO-8859-1")

d2['label']=1
data1=pd.concat([d1,d2],axis=0)

d1=pd.read_csv('ot_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0

d2=pd.read_csv('ot_non_rumour.csv',header=0,encoding = "ISO-8859-1")

d2['label']=1


data2=pd.concat([d1,d2],axis=0)


d1=pd.read_csv('sydney_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0

d2=pd.read_csv('sydney_non_rumour.csv',header=0,encoding = "ISO-8859-1")

d2['label']=1
data3=pd.concat([d1,d2],axis=0)


d1=pd.read_csv('charlie_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0

d2=pd.read_csv('charlie_non_rumour.csv',header=0,encoding = "ISO-8859-1")

d2['label']=1
data4=pd.concat([d1,d2],axis=0)

data=pd.concat([data1,data2,data3,data4],axis=0)

print(len(data))


###########
#Testing
##########

d1=pd.read_csv('fergu_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0

d2=pd.read_csv('fergu_non_rumour.csv',header=0,encoding = "ISO-8859-1")

d2['label']=1
td=pd.concat([d1,d2],axis=0)



print(len(td))


train_data=data['text']
train_target=data['label']

test_data=td['text']
test_target=td['label']





x_train, x_val, y_train, y_val = train_test_split(train_data,train_target, test_size=0.2, random_state=62,shuffle=True)
print("total train:{} val:{} test:{}".format(len(x_train),len(x_val),len(test_data)))
print("\n")


train=pd.DataFrame({'text':x_train,'label':y_train})  
rumour_train=train.loc[train['label'] == 0]
all_words =rumour_train['text'].str.split(expand=True).unstack().value_counts()
tracker=all_words.index.values[2:200] # terms in top X terms
counts=all_words.values[2:200] # Occurrences of each term in top X
#coverting to list for programing simplicity
tracker=list(tracker)
counts=list(counts)
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stop_words=stopWords

for i_t in tracker:
    for s in stop_words:
        if i_t==s:
            idx=tracker.index(i_t)
            tracker.pop(idx)
            counts.pop(idx)
# Getting the summation of occurrences of all top 50 terms-- Denominator of point 4.1
sumv=0
for item in counts:
    sumv=sumv+item
total_counts=sumv 

vec= TfidfVectorizer(stop_words='english',max_features=1500)
x_trainvec=vec.fit_transform(x_train)
 #List for Holding accuracy score for different K value of Logistic Regression
logistic_auc=[] #... Holding AUC score for Logistic
logistic_f1_score=[]
logistic_avg_precision=[]
logistic_precision_score=[]
logistic_recall_score=[]


svm_auc=[] #... Holding AUC score for Logistic
svm_f1_score=[]
svm_avg_precision=[]
svm_precision_score=[]
svm_recall_score=[]

k_l=[]
k_vals=[]
for i in range(0,100):
    k_vals.append(i)
    
for k in k_vals:
    x_valvec=vec.transform(x_val)
    feat=vec.get_feature_names()
    for f in feat:
        for item in tracker:
            if f==item:
                #getting the value to be added
                f_index=tracker.index(f)
                count=counts[f_index]
                value=count/total_counts
               
                for i in range(0,x_valvec.shape[0]):
                    idx=feat.index(f)
                    if x_valvec[i,idx]>0:
                        
                        #print("\n Before:{}".format(x_valvec[i,idx]))
                        x_valvec[i,idx]+=(value*k)
                        #print(" After:{}".format(x_valvec[i,idx]))


    #Now doing classification with logistic regression
    clf=linear_model.LogisticRegression()
    clf.fit(x_trainvec,y_train)
    pred=clf.predict(x_valvec)
  
    #auc
    auc=roc_auc_score(y_val,pred)
    logistic_auc.append(auc)# Only used in plotting
    avp=average_precision_score(y_val,pred)
    logistic_avg_precision.append(avp) #only used in plotting
    f1=precision_score(y_val,pred)
    logistic_precision_score.append(f1)
    f1=recall_score(y_val,pred)
    logistic_recall_score.append(f1)
    f1=f1_score(y_val,pred)
    logistic_f1_score.append(f1)
    
    #NSVM
    clf=SVC(kernel='linear')
    clf.fit(x_trainvec,y_train)
    pred=clf.predict(x_valvec)
  
    #auc
    auc=roc_auc_score(y_val,pred)
    svm_auc.append(auc)# Only used in plotting
    avp=average_precision_score(y_val,pred)
    svm_avg_precision.append(avp) #only used in plotting
    f1=precision_score(y_val,pred)
    svm_precision_score.append(f1)
    f1=recall_score(y_val,pred)
    svm_recall_score.append(f1)
    f1=f1_score(y_val,pred)
    svm_f1_score.append(f1)
    
    
    
    #only used in plotting
    
    k_l.append(k)
    


max_log_auc=max(logistic_auc)
max_log_avp=max(logistic_avg_precision)
max_log_f1=max(logistic_f1_score)
max_log_precision=max(logistic_precision_score)
max_log_recall=max(logistic_recall_score)
#Best K Val for logistic interms of AUC and AVP
k_log_auc=k_vals[logistic_auc.index(max_log_auc)]
k_log_avp=k_vals[logistic_avg_precision.index(max_log_avp)]
k_log_f1=k_vals[logistic_f1_score.index(max_log_f1)]
k_log_precision=k_vals[logistic_precision_score.index(max_log_precision)]
k_log_recall=k_vals[logistic_recall_score.index(max_log_recall)]





max_log_auc=max(svm_auc)
max_log_avp=max(svm_avg_precision)
max_log_f1=max(svm_f1_score)
max_log_precision=max(svm_precision_score)
max_log_recall=max(svm_recall_score)
#Best K Val for logistic interms of AUC and AVP
k_svm_auc=k_vals[svm_auc.index(max_log_auc)]
k_svm_avp=k_vals[svm_avg_precision.index(max_log_avp)]
k_svm_f1=k_vals[svm_f1_score.index(max_log_f1)]
k_svm_precision=k_vals[svm_precision_score.index(max_log_precision)]
k_svm_recall=k_vals[svm_recall_score.index(max_log_recall)]



#################
#TESTING WITH BOOSTING
#################
train=pd.DataFrame({'text':train_data,'label':train_target})
rumour_train=train.loc[train['label'] == 0]
all_words =rumour_train['text'].str.split(expand=True).unstack().value_counts()
tracker=all_words.index.values[2:200] # terms in top X terms
counts=all_words.values[2:200] # Occurrences of each term in top X
tracker=list(tracker)
counts=list(counts)
for i_t in tracker:
    for s in stop_words:
        if i_t==s:
            idx=tracker.index(i_t)
            tracker.pop(idx)
            counts.pop(idx)

# Getting the summation of occurrences of all top 50 terms-- Denominator of point 4.1
sumv=0
for item in counts:
    sumv=sumv+item
total_counts=sumv 
vec= TfidfVectorizer(stop_words='english',max_features=1500)
train_datavec=vec.fit_transform(train_data)

train_datavec_log=vec.fit_transform(train_data)
train_datavec_svm=vec.fit_transform(train_data)
x_test=vec.transform(test_data)


##################
#F1
##################
train_datavec_log=vec.fit_transform(train_data)
train_datavec_svm=vec.fit_transform(train_data)
x_test_log=vec.transform(test_data)  #vectorized Logsitic 
x_test_svm=vec.transform(test_data) # vectorized SVM
feat=vec.get_feature_names()
for f in feat:
    for item in tracker:
        if f==item:
            #getting the value to be added
            f_index=tracker.index(f)
            count=counts[f_index]
            value=count/total_counts
               
            for i in range(0,x_test.shape[0]):
                idx=feat.index(f)
                
                if x_test[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    x_test_log[i,idx]+=(value*k_log_f1)
                    x_test_svm[i,idx]+=(value*k_svm_f1)
            for i in range(0,train_datavec.shape[0]):
                idx=feat.index(f)
                if train_datavec[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    train_datavec_log[i,idx]+=(value*k_log_f1)
                    train_datavec_svm[i,idx]+=(value*k_svm_f1)

print("\n After Boosting checking the F1 score with std classifier")
#Lr
clf=linear_model.LogisticRegression()
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR  F1:{}  ".format(f1_score(test_target,pred)))

clf=SVC(kernel='linear')
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with SVMF1:{}  ".format(f1_score(test_target,pred)))



clf=linear_model.LogisticRegression(C=10)
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR  Tuned F1:{}  ".format(f1_score(test_target,pred)))

clf=SVC(kernel='linear',C=5)
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with Tuned SVM:{}  ".format(f1_score(test_target,pred)))













################3
#Precision
################
train_datavec_log=vec.fit_transform(train_data)
train_datavec_svm=vec.fit_transform(train_data)
x_test_log=vec.transform(test_data)  #vectorized Logsitic 
x_test_svm=vec.transform(test_data) # vectorized SVM
feat=vec.get_feature_names()
for f in feat:
    for item in tracker:
        if f==item:
            #getting the value to be added
            f_index=tracker.index(f)
            count=counts[f_index]
            value=count/total_counts
               
            for i in range(0,x_test.shape[0]):
                idx=feat.index(f)
            
                if x_test[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    x_test_log[i,idx]+=(value*k_log_precision)
                    x_test_svm[i,idx]+=(value*k_svm_precision)
            for i in range(0,train_datavec.shape[0]):
                idx=feat.index(f)
                if train_datavec[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    train_datavec_log[i,idx]+=(value*k_log_precision)
                    train_datavec_svm[i,idx]+=(value*k_svm_precision)

print("\n After Boosting checking the PR score with std classifier")
#Lr
clf=linear_model.LogisticRegression()
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR  P:{} ".format(precision_score(test_target,pred)))

clf=SVC(kernel='linear')
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with SVM  P:{}  ".format(precision_score(test_target,pred)))


clf=linear_model.LogisticRegression(C=10)
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR Tuned  P:{} ".format(precision_score(test_target,pred)))

clf=SVC(kernel='linear',C=5)
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with SVM Tuned  P:{}  ".format(precision_score(test_target,pred)))

#################
#Recall
############3
train_datavec_log=vec.fit_transform(train_data)
train_datavec_svm=vec.fit_transform(train_data)
x_test_log=vec.transform(test_data)  #vectorized Logsitic 
x_test_svm=vec.transform(test_data) # vectorized SVM
feat=vec.get_feature_names()
for f in feat:
    for item in tracker:
        if f==item:
            #getting the value to be added
            f_index=tracker.index(f)
            count=counts[f_index]
            value=count/total_counts
               
            for i in range(0,x_test.shape[0]):
                idx=feat.index(f)
                
                if x_test[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    x_test_log[i,idx]+=(value*k_log_recall)
                    x_test_svm[i,idx]+=(value*k_svm_recall)
            for i in range(0,train_datavec.shape[0]):
                idx=feat.index(f)
                if train_datavec[i,idx]>0:
                        
                    #print("\n Before:{}".format(x_valvec[i,idx]))
                    train_datavec_log[i,idx]+=(value*k_log_recall)
                    train_datavec_svm[i,idx]+=(value*k_svm_recall)

print("\n After Boosting checking the Recall score with std classifier")
#Lr
clf=linear_model.LogisticRegression()
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR R:{} ".format(recall_score(test_target,pred)))

clf=SVC(kernel='linear')
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with SVM R:{} ".format(recall_score(test_target,pred)))


clf=linear_model.LogisticRegression(C=10)
clf.fit(train_datavec_log,train_target)
pred=clf.predict(x_test_log)

print("LR Tuned R:{} ".format(recall_score(test_target,pred)))

clf=SVC(kernel='linear',C=5)
clf.fit(train_datavec_svm,train_target)
pred=clf.predict(x_test_svm)

print("with SVM Tuned R:{} ".format(recall_score(test_target,pred)))



