#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("Spam Data.csv")
inputs = df.drop("Spam", axis="columns")
target = df["Spam"]


# In[8]:


inputs.head()


# In[27]:


from sklearn.model_selection import train_test_split
x_tr, x_te, Y_tr, Y_te = train_test_split(inputs, target, test_size=0.2, random_state=42)


# In[28]:


from sklearn import tree
modle = tree.DecisionTreeClassifier()
modle.fit(x_tr, Y_tr)


# In[32]:


modle.score(x_te, Y_te)


# In[33]:


df.head()


# In[ ]:





# In[ ]:




