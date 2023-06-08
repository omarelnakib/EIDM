#!/usr/bin/env python
# coding: utf-8

# In[1]:



from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Embedding
from tensorflow.keras.layers import LSTM, Conv1D,Conv2D, MaxPooling1D, Flatten,Bidirectional

from tensorflow.keras.optimizers import RMSprop, SGD
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, normalize, LabelEncoder
from sklearn.model_selection import train_test_split

# In[30]:


# In[2]:




#read dataset
data= pd.read_csv("./divided 6 attacks/ResampledData.csv").replace([np.inf, -np.inf], np.nan).dropna(how="any")
#split dataset to train and test

labels = data.pop('Label')

#print(labels)
pdtrain_data, pdtest_data,pdtrain_Label,pdtest_label = train_test_split(data,labels, test_size=0.2,shuffle=True,stratify=labels)

# 

#convert data from pd to numpy arrary
train_data = pdtrain_data.to_numpy()
test_data = pdtest_data.to_numpy()

            
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')


#normalize data
train_data= normalize(train_data, axis=1, norm='l1')
test_data= normalize(test_data, axis=1, norm='l1')



# In[3]:



#reshape data to 3D
train_data=train_data.reshape(train_data.shape[0],train_data.shape[1])
test_data=test_data.reshape(test_data.shape[0],test_data.shape[1])



#feature selection
test = SelectKBest(score_func=f_classif, k=78)
fit = test.fit(train_data, pdtrain_Label)
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(train_data)

# Summarize selected features


# In[4]:



train_data = features
print(test_data.shape)
test_data = fit.transform(test_data)


# In[5]:



# In[31]:



label_encoder = LabelEncoder()
#convert string labels to int values to be able to classify them
trainVec = label_encoder.fit_transform(pdtrain_Label)
testVec = label_encoder.fit_transform(pdtest_label)

#categorize labels
train_labels = keras.utils.to_categorical(trainVec, 6)
test_labels = keras.utils.to_categorical(testVec, 6)


# In[32]:


# In[6]:


print(labels)


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:

# reshape data to 3D
train_data=train_data.reshape(train_data.shape[0],train_data.shape[1],1)
test_data=test_data.reshape(test_data.shape[0],test_data.shape[1],1)
print(train_data.shape)

#prepare MLP Model
model = Sequential()

model.add(Dense(120,activation='relu'))
model.add(Conv1D(80, 20, activation='relu'))
# model.add(Conv1D(60, 10, activation='relu'))
model.add(MaxPooling1D())
# Dropout to avoid overfitting
model.add(Dropout(0.2))
# Flatten the results to one dimension for passing into our next layer
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))

model.add(Dense(6, activation='sigmoid'))
# In[33]:


#opt = RMSprop(lr = 0.008, momentum=0.9)
opt = keras.optimizers.Adam(learning_rate=0.0005)
# opt = SGD(lr=0.05,momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(train_data.shape)
print(train_labels.shape)

print(test_data.shape)
print(test_labels.shape)

history = model.fit(train_data, train_labels,
                    batch_size=3000,
                    epochs=1,
                    validation_data=(test_data, test_labels))


# In[ ]:


score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# In[ ]:

predict = model.predict_classes(test_data,verbose=1)
print(predict);






# In[17]:


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report

ptestVec = label_encoder.fit_transform(pdtest_label)

print(ptestVec);

print(confusion_matrix(ptestVec, predict))
print(classification_report(ptestVec, predict, digits=6))


# In[ ]:





# In[ ]:




