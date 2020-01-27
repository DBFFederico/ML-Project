#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras import regularizers


# In[45]:


#for upload

upload_written= np.load('written_test.npy', allow_pickle=True)
upload_spoken= np.load('spoken_test.npy', allow_pickle= True)


# In[6]:


#load dataset
spoken= np.load('spoken_train.npy', allow_pickle= True)
written= np.load('written_train.npy', allow_pickle= True)
match= np.load('match_train.npy', allow_pickle= True)


# In[14]:


#Load balanced datasets
spoken_b= np.load('balanced train/balan_spoken.npy', allow_pickle=True)
written_b= np.load('balanced train/balan_written.npy', allow_pickle=True)
match_b= np.load('balanced train/balan_match.npy', allow_pickle=True)


# In[7]:


w_train= np.load('written_train.npy', allow_pickle=True)
s_train= np.load('spoken_train.npy', allow_pickle=True)

w_test= np.load('written_test.npy', allow_pickle=True)
s_test= np.load('spoken_test.npy', allow_pickle=True)

match= np.load('match_train.npy', allow_pickle=True)


# In[8]:


#create hard copies
match2 =match.copy()
w_train2= w_train.copy()
s_train2= s_train.copy()


# In[9]:


#Retrieve the indeces of false and true
true_lab_ind= np.where(match2== True)
false_lab_ind= np.where(match2== False)


# In[10]:


#create balance match array
true_lab= match2[true_lab_ind]
false_lab= match2[false_lab_ind]
type(true_lab)
false_lab= false_lab[:len(true_lab)]
print(len(true_lab))
print(len(false_lab))
balan_match= np.concatenate((true_lab,false_lab))

np.save('balan_match', balan_match)


# In[11]:


#create balance written train
w_true= w_train2[true_lab_ind]
w_false= w_train2[false_lab_ind]

w_false= w_false[: len(true_lab)]
w_false.shape

balan_written= np.concatenate((w_true, w_false), axis=0)
balan_written.shape
np.save('balan_written', balan_written)


# In[12]:


#create a balnced spoken train
s_true= s_train2[true_lab_ind]
s_false= s_train2[false_lab_ind]

s_false= s_false[: len(true_lab)]
s_false.shape

balan_spoken= np.concatenate((s_true, s_false), axis=0)
balan_written.shape
np.save('balan_spoken', balan_spoken)


# In[15]:


#Standard scaling spoken 
def scaling_spoken(array):
    
    stc= StandardScaler()
    stc.fit(array[0])
    scaled_spoken= np.array([stc.transform(x) for x in array])
    return scaled_spoken, stc
scaled, stc= scaling_spoken(spoken_b)


# In[16]:


#all data scaling spoken
scaled_all=np.array([stc.transform(x) for x in spoken])


# In[46]:


#scaled upload

scaled_upload= np.array([stc.transform(x) for x in upload_spoken])


# In[17]:


#padding function
def flat_zero(array):
    '''Return array to concatenate with the written'''
    flatted= [x.flatten() for x in array]
    lenght= [len(list(x)) for x in flatted]
    a=[]
    for i in flatted:
        #diff= max(lenght) - len(list(i))
        diff= max(lenght) - len(list(i))
        a.append(np.concatenate([i, np.zeros(diff)]))
    return a


# In[47]:


#zero padding balanced spoken
zeros_PCA= np.array(flat_zero(scaled))
#zero padding all spoken
zeros_PCA_all= np.array(flat_zero(scaled_all))

#zeros upload
zeros_PCA_upload= np.array(flat_zero(scaled_upload))


# In[19]:


#PCA decomposition function
def PCA_decop(array):
    '''PCA transformation on scaled array'''
    pca= PCA(.80)
    
    transpose= [x.T for x in array]
    #pca.fit(array[0])
    #transformed =[pca.transform(x) for x in array]
    #transformed2 =[x.T for x in transformed]
    transformed2= pca.fit_transform(array)
    return transformed2, pca


# In[48]:


#PCA audio balanced
transformed, pca = PCA_decop(zeros_PCA)
transformed= np.array(transformed)

#PCA audio all
transformed_all = pca.transform(zeros_PCA_all)

#PCA upload
pca_upload = pca.transform(zeros_PCA_upload)


# In[49]:


#WRITTEN PREPROCESSING

#scaling PCA balanced written
stw= StandardScaler()
written_s= stw.fit_transform(written_b)
pca_w= PCA(.80)
written_t= pca_w.fit_transform(written_s)

#scaling PCA all written
written_s_all= stw.transform(written)
written_t_all= pca_w.transform(written_s_all)

#scaling PCA written upload
written_scaled_upload= stw.transform(upload_written)
written_pca_upload= pca_w.transform(written_scaled_upload)


# In[50]:


#Concatenation of balanced arrays
X_train_bal= np.column_stack([transformed, written_t])

#Concatenation all data
X_train_all= np.column_stack([transformed_all, written_t_all])

#Concatenation upload data
X_test_upload= np.column_stack([pca_upload, written_pca_upload])


# In[40]:


#Encoding balanced target
onehot = LabelBinarizer()
Y_train_bal= onehot.fit_transform(match_b)

#Encoding all target
Y_train_all= onehot.transform(match)


# In[41]:


#Splitting balanced data
X_train, X_test, y_train, y_test = train_test_split(X_train_bal ,Y_train_bal, test_size=0.10, random_state=666)

#Splitting all data

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_train_all, Y_train_all, test_size=0.30, random_state=666)


# In[42]:


#NEURAL NETWORK
optimizer = Adam(lr=0.001)
model = Sequential()
model.add(Dense(280, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 


model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(X_train, y_train, epochs=80, batch_size=32, verbose=2)


# In[43]:


#balanced test
from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test, verbose=2)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))


# In[44]:


#all test
y_pred_all = model.predict_classes(X_test_all, verbose=2)
print(accuracy_score(y_pred_all, y_test_all))
print(confusion_matrix(y_test_all, y_pred_all))


# In[51]:


#Predictions
y_pred_upload = model.predict_classes(X_test_upload, verbose=2)

