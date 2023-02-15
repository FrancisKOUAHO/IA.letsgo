#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

ratings = pd.read_csv('data/ratings.csv')
ratings.head()


# In[5]:


user_item = ratings.groupby(['userId', 'activityId'])['rate'].first().unstack(fill_value=0.0)
user_item.shape


# In[6]:


user_item.shape


# In[7]:


user_item.describe


# In[8]:


user_item.loc[42].sort_values(ascending=False).head()
# Trie des valeur pour l'user_item 42


# In[12]:


U, sigma, Vt = svds(user_item.to_numpy(), k=50)
#U, sigma et VT sont des matrices
#svds décompose partiellement en valeurs singulières d'une matrice creuse.
U.shape


# In[13]:


Vt.shape


# In[15]:


sigma_diag_matrix=np.diag(sigma)
# La méthode diag sert a créer / extraire une diagonale


# In[16]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)
# Ici la variable all_user est un produit scalaire des matrices / diagonale, la methode dot est la pour
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = user_item.columns, index=user_item.index)
# On crée un Dataframe


# In[17]:


preds_df.shape


# In[35]:


preds_df.head()


# In[37]:


user_item.loc[42].sort_values(ascending=False).head(10)


# In[39]:


activities_user_42 = user_item.loc[42]


# In[40]:


high_rated_activities_42 = activities_user_42[activities_user_42 > 3].index


# In[41]:


high_rated_activities_42


# In[42]:


activities_recommended_for_42 = preds_df.loc[42]


# In[44]:


activities_high_recommend_for_42 = activities_recommended_for_42[activities_recommended_for_42 > 3].index


# In[45]:


activities_high_recommend_for_42


# In[46]:


set(activities_high_recommend_for_42) - set(high_rated_activities_42)


# In[66]:


def get_high_recommended_activities(userId):
    activities_rated_by_user = user_item.loc[userId]
    activities_high_rated_by_user =  activities_rated_by_user[activities_rated_by_user > 3].index
    activities_recommended_for_user = preds_df.loc[userId]
    activities_high_recommend_for_user = activities_recommended_for_user[activities_recommended_for_user > 3].index
    res = dict()
    res["activityId"] = set(activities_high_recommend_for_user) - set(activities_high_rated_by_user)
    res["rate"] = preds_df.loc[userId, set(activities_high_recommend_for_user) - set(activities_high_rated_by_user)]
    print(res)
    return res