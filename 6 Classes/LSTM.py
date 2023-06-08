#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[35]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten,LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, normalize, LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
#read dataset
data= pd.read_csv("./divided 6 attacks/ResampledData.csv").replace([np.inf, -np.inf], np.nan).dropna(how="any")
labels = data.pop('Label')








# In[3]:


# function to get unique values
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list;
   
     
   


# In[4]:


l = unique(labels)
print(l)
list(labels)

#encode labels
label_encoder = LabelEncoder()
label_encoder.fit(l)
print(label_encoder.classes_)
labels = label_encoder.transform(labels)

# labels = label_encoder.fit_transform(labels)


# In[5]:





#split dataset to train and test
pdtrain_data, pdtest_data, pdtrain_Label, pdtest_label= train_test_split(data,labels, test_size=0.2, shuffle=True, stratify = labels)


#convert data from pd to numpy arrary
train_data = pdtrain_data.to_numpy()
test_data = pdtest_data.to_numpy()

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

#normalize data
train_data= normalize(train_data, axis=1, norm='l1')
test_data= normalize(test_data, axis=1, norm='l1')

#categorize labels
train_labels = keras.utils.to_categorical(pdtrain_Label, 6)
test_labels = keras.utils.to_categorical(pdtest_label, 6)
print(pdtrain_Label)
print(train_labels)


# In[38]:




# In[6]:




#feature selection
test = SelectKBest(score_func=f_classif, k=78)
fit = test.fit(train_data, pdtrain_Label)
np.set_printoptions(precision=3)

features = fit.transform(train_data)
train_data = features
print(test_data.shape)
test_data = fit.transform(test_data)


#reshape data to 3D
train_data=train_data.reshape(train_data.shape[0],train_data.shape[1],1)
test_data=test_data.reshape(test_data.shape[0],test_data.shape[1],1)
print(train_data.shape)


# In[7]:



input_shape = (78,1)


#prepare MLP Model
model = Sequential()
model.add(Dense(64, activation='relu'))

model.add(LSTM(64, dropout=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(6, activation='sigmoid'))

#model.summary()
opt = SGD(lr=0.01,momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


history = model.fit(train_data, train_labels,
                    batch_size=1100,
                    epochs=1,
                    validation_data=(test_data, test_labels))



# In[ ]:



predict = model.predict_classes(test_data,verbose=1)
print(predict);
(loss, accuracy, precision, recall) = model.evaluate(test_data, test_labels, verbose=1)


# In[16]:


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report

print(pdtest_label);
print(confusion_matrix(pdtest_label, predict))
print(classification_report(pdtest_label, predict, digits=6))


# In[ ]:




