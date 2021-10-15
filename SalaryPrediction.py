#!/usr/bin/env python
# coding: utf-8

# # **SOFTWARE DEVELOPER SALARY PREDICTION**

# Dataset source: https://insights.stackoverflow.com/survey

# In[1]:


# install libraries and load dataset
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("survey_results_public.csv")


# In[ ]:





# In[2]:


# check initial 5 rows of data
df.head()


# In[3]:


# lets check column name and count
print('Total Number of column;',len(list(df.columns))) # this no. of column
print('---------------------')
print('Name of column:',list(df.columns)) # to get the column names 


# out 61 we  will just 5 columns, This thing has been decided after complete study of each every col. then we came to this conclusion.
# 

# In[4]:


# select import features only & we drop remaining col.
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]] # to get required col.
df = df.rename({"ConvertedComp": "Salary"}, axis=1) # change in col. name
df.head()


# In[5]:


# lets check target without Nan
df = df[df["Salary"].notnull()]
df.head()


# In[6]:


# information about dataframe/dataset
df.info()


# In[7]:


# we will drop Nan Values row 
df = df.dropna()
df.isnull().sum()


# In[8]:


# information about dataframe/dataset
df.info()


# In[9]:


# we try to keep full employment details only
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[10]:


# lets check which country has more datapoints
df['Country'].value_counts()


# In[11]:


# create fuction for keep countries which have above cutoff and below cutoff add into other cat.
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[12]:


# lets rut the above function and map with present col.
country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[13]:


# lets inspect salary range
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# important: from above box we can see that we have large amount of outliers, we will keep the data which has salary within the range (000 t0 250000 )

# In[14]:


# keep min salary above 10000 and max salary 250000
df = df[df["Salary"] <= 250000] # max 
df = df[df["Salary"] >= 10000] # min
df = df[df['Country'] != 'Other'] # drop others


# In[15]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[16]:


# check unique value in experience 
df["YearsCodePro"].unique()


# In[17]:


# lets clean and replace less than 1  years & more the 50 year.
def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)


# In[18]:


# agian check unique in experaince 
df["YearsCodePro"].unique()


# In[19]:


# check uniques things in educations
df["EdLevel"].unique()


# In[20]:


# clean and split into bachelers, masters, Post Grad & less than grad.
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)


# In[21]:


# again check education
df["EdLevel"].unique()


# In[22]:


# lets convert string into number (ordinal encoding)
from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()
#le.classes_


# In[23]:


# lets convert string into number (ordinal encoder)
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()


# In[24]:


# lets again check datafram
df


# In[ ]:





# In[25]:


# lets assgin input features and target
X = df.drop("Salary", axis=1) # input
y = df["Salary"]   # target


# ### Linear Regression Model 

# In[26]:


# lets import linear regression model
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)


# In[27]:


# lets do the prediciton
y_pred = linear_reg.predict(X)


# In[28]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[ ]:





# ### Decision Tree Regressor Model 

# In[29]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0) # we can use max depth
dec_tree_reg.fit(X, y.values)


# In[30]:


y_pred = dec_tree_reg.predict(X)


# In[31]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# ### Random Forest Regressor Model 

# In[32]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)


# In[33]:


y_pred = random_forest_reg.predict(X)


# In[34]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# ### GridSearchCV model

# In[35]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)


# In[36]:


regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[37]:


X


# In[38]:


# suppose we have new array
# country, edlevel, yearscode
X = np.array([["United States", 'Master’s degree', 15 ]])


# In[39]:


X


# In[40]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[41]:


y_pred = regressor.predict(X)
y_pred


# In[42]:


# import library forsave model
import pickle


# In[43]:


# save model wb (write binary mode)
data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps_01.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[44]:


# lets load the saved model
with open('saved_steps_01.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[45]:


y_pred = regressor_loaded.predict(X)
y_pred


