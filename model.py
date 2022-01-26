#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


data=pd.read_csv("iris (3).csv")


# In[8]:


data


# In[9]:


data.isnull().sum()


# In[10]:


import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[12]:


sns.pairplot(data)


# In[13]:


array = data.values
X = array[:,0:4]    
Y = array[:,4]  


# In[14]:


validation_size = 0.20
seed = 41
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[41]:


model = LogisticRegression()


# In[42]:


model.fit(X_train,Y_train)


# In[43]:


model.predict(X_test)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[45]:


knn = KNeighborsClassifier()
dtree = DecisionTreeClassifier()
svm = SVC()


# In[46]:


knn.fit(X_train, Y_train)
print("accuracy :" , knn.score(X_test,Y_test))


# In[47]:


dtree.fit(X_train, Y_train)
print("accuracy :" , dtree.score(X_test,Y_test))


# In[48]:


svm.fit(X_train, Y_train)
print("accuracy :" , svm.score(X_test,Y_test))


# In[50]:


import pickle
pickle.dump(model,open('model.pkl','wb'))


# In[ ]:





# In[ ]:




