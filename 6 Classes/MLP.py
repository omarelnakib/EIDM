
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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


#encode labels
label_encoder = LabelEncoder()
label_encoder.fit(l)

labels = label_encoder.transform(labels)


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

#train_labels = pd.Categorical(pdtrain_Label)
#test_labels = pd.Categorical(pdtest_label)




# In[6]:



#feature selection
test = SelectKBest(score_func=f_classif, k=78)
fit = test.fit(train_data, pdtrain_Label)

train_data = fit.transform(train_data)

test_data = fit.transform(test_data)


# In[7]:



from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report


#prepare MLP Model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(78,)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='sigmoid'))
model.summary()

opt = SGD(learning_rate=0.08, momentum=0.4)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=(['accuracy']))


history = model.fit(train_data, train_labels,
                     batch_size=10000,
                     epochs=100,
    verbose=1,
                    validation_data=(test_data, test_labels),
                   )



predict=model.predict(test_data) 
predict=np.argmax(predict,axis=1)

(loss, accuracy, precision, recall) = model.evaluate(test_data, test_labels, verbose=1)


# In[9]:


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report

print(confusion_matrix(pdtest_label, predict))
print(classification_report(pdtest_label, predict, digits=6))


# In[51]:





