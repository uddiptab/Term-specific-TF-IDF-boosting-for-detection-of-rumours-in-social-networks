from lstm import *
import pandas as pd
import numpy as np
import gc
import os
import nltk
import tqdm
import numpy as np
import pandas as pd
#nltk.download("punkt")
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten
from keras.layers import concatenate, GRU, Input, K, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
import tensorflow as tf
from sklearn.metrics import *
np.random.seed(0)



print(" Necessary Libraries Imported")

print(" Necessary Libraries Imported")
print("\n Loading Data...")
# Appending Labels to Rumour and Non Rumour Data
#rumour = Label 0 , Non-Rumour =Label 1
d1=pd.read_csv('fergu_rumour.csv',header=0,encoding = "ISO-8859-1")
d1['label']=0
d2=pd.read_csv('fergu_non_rumour.csv',header=0,encoding = "ISO-8859-1")
d2['label']=1
data=pd.concat([d1,d2],axis=0)
print(" Data loading Completed")


data=pd.concat([data['text'],data['label']],axis=1)
data_=data['text']
label_=data['label']
data_.fillna('NAN')

# embedding_path = "../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec"
embedding_path = "../../../Capsule-Rumour/glove.txt"

batch_size = 128 # 256
recurrent_units = 16 # 64
dropout_rate = 0.3 
dense_size = 8 # 32
sentences_length =  28


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
CLASSES = ["project_is_approved"]




auc_list=[]
fp_list=[]
f1_list=[]
avp_list=[]
prec_list=[]
recall_list=[]

fold=1
for i in range(0,fold):
    
    print("Loading data...")
    train_data, test_data, train_target, test_target = train_test_split(data_,label_, test_size=0.2, random_state=42,shuffle=True)
    
    
    
    list_sentences_train = train_data
    list_sentences_test = test_data
    y_train = train_target
    
    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)
    
    
    # Embedding
    words_dict[UNKNOWN_WORD] = len(words_dict)
    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(embedding_path)
    embedding_size = len(embedding_list[0])
    
    
    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)
    
    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)
    
    embedding_matrix = np.array(embedding_list)
    
    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        sentences_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)
    
    get_model_func = lambda: get_model(
        embedding_matrix,
        sentences_length,
        dropout_rate,
        recurrent_units,
        dense_size)
    
    del train_data, test_data, list_sentences_train, list_sentences_test
    del tokenized_sentences_train, tokenized_sentences_test, words_dict
    del embedding_list, embedding_word_dict
    del train_list_of_token_ids, test_list_of_token_ids
    gc.collect();
    
    
    
    print(embedding_matrix.shape, sentences_length,X_train.shape)
    print("Starting to train models...")
    model = get_model(embedding_matrix,
        sentences_length,
        dropout_rate,
        recurrent_units,
        dense_size)
    
    
    batch_size =128
    epochs = 500
    import time
    start= time.time()
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=20,verbose=1)
    #hist.summary()
    
    y_pred = model.predict(X_test, verbose=1)
    end= time.time()
    
    from sklearn.metrics import *
    from sklearn.metrics import roc_auc_score
    tn, fp, fn, tp = confusion_matrix(test_target,y_pred.round()).ravel()
    print("The Accuracy:{}  AUC{} F1 Score:{} Precision:{} Recall:{} FP:{}".format(accuracy_score(test_target,y_pred.round()),roc_auc_score(test_target,y_pred),f1_score(test_target,y_pred.round()),precision_score(test_target,y_pred.round()),recall_score(test_target,y_pred.round()),fp/(fp+tn)))
    auc_list.append(roc_auc_score(test_target,y_pred))
    fp_list.append(fp/(fp+tn))
    f1_list.append(f1_score(test_target,y_pred.round()))
    avp_list.append(average_precision_score(test_target,y_pred.round()))
    prec_list.append(precision_score(test_target,y_pred.round()))
    recall_list.append(recall_score(test_target,y_pred.round()))
    




print("\n\n\n ------------Model Statistics---------------")

#AUC
sumauc=0
for item in auc_list:
    sumauc=sumauc+item
mean_auc=sumauc/fold

#F1-Score
sumf1=0
for item in f1_list:
    sumf1=sumf1+item
mean_f1=sumf1/fold

#FP-Rate
sumfp=0
for item in avp_list:
    sumfp=sumfp+item
mean_avp=sumfp/fold

sump=0
for item in prec_list:
    sump=sump+item
mean_prec=sump/fold

sumr=0
for item in recall_list:
    sumr=sumr+item
mean_rec=sumr/fold






print("Number of epochs: {}".format(200))
print("Batch Size: {}".format(128))
print("Average AUC score over 10 folds : {}%".format(mean_auc*100))

print("Average AVP over 10 folds : {}%".format(mean_avp*100))

print("Average  P over 10 folds : {}%".format(mean_prec*100))


print("Average R over 10 folds : {}%".format(mean_rec*100))
print("Time taken:{}".format(end-start))


def result():
    return y_pred,mean_auc*100,mean_f1*100