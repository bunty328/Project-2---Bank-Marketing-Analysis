#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[80]:


import warnings
warnings.filterwarnings("ignore")


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the dataset

# In[3]:


input1 = pd.read_csv(r"D:\Data Science\UpGrad\Exploratory Data Analysis\Data Cleaning\Bank Dataset\bank_marketing_updated_v1.csv")
input1.head()


# ## Data Cleaning

# In[4]:


input1 = pd.read_csv(r"D:\Data Science\UpGrad\Exploratory Data Analysis\Data Cleaning\Bank Dataset\bank_marketing_updated_v1.csv", skiprows = 2)
input1.head()


# ### Droping customerid Column

# In[5]:


input1.drop("customerid", axis = 1, inplace = True)
input1.head()


# #### Dividing jobedu into two columns i.e Job & Education

# In[6]:


input1['Job'] = input1.jobedu.apply(lambda x: x.split(",")[0])
input1.head()


# In[7]:


input1['Education'] = input1.jobedu.apply(lambda x: x.split(",")[1])
input1.head()


# In[8]:


input1.drop("jobedu", axis = 1, inplace = True)
input1.head()


# #### Extrat the value of Month from column 'month'

# In[9]:


input1['Month'] = input1.month.apply(lambda x: x.split(",")[0])
input1.head()


# In[11]:


#check if x is float or not
input1[input1.month.apply(lambda x: isinstance(x, float)) == True]


# As In the above data set, I can see so many missing values.

# ### Checking for missing values

# In[10]:


input1.isnull().sum()


# ## Handling missing values

# In[12]:


input1.age.isnull().sum()


# In[13]:


input1.shape


# In[14]:


100*20/45211


# #### Drop records with age missing

# In[15]:


input2 = input1[~input1.age.isnull()].copy()
input2.shape


# In[16]:


input2.age.isnull().sum()


# #### Handling missing values with months 

# In[17]:


input2.month.isnull().sum()


# In[18]:


input2.month.value_counts(normalize = True)


# In[19]:


month_mode = input1.month.mode()[0]
month_mode


# In[ ]:


input2.month = input2.month.fillna(month_mode)


# In[20]:


input2.month.fillna(month_mode, inplace = True)
input2.month.value_counts(normalize = True)


# In[21]:


input2.month.isnull().sum()


# In[22]:


input2.pdays.describe()


# #### -1 indicates the missing values !!

# ### How do we handle null values with this?
# #### Remember our objective!
#  - We want the missing value to be ignored in the calucaltions.
#  - Simply make it missing - replace -1 with NaN
#  - all sammary statistics - mean, median etc. will ignore the missing values.

# In[23]:


input2.loc[input2.pdays<0,"pdays"] = np.NaN
input2.pdays.describe()


# Missing values doesn't always have to be Null!

# In[24]:


input2.response.isnull().sum()


# In[25]:


input2.shape


# In[26]:


100*30/45191


# ## or

# In[27]:


float(100.0*(input2.response.isnull().sum())//(input2.shape[0]))


# # Outlier handling

# #### Age Variable

# In[28]:


input2.age.describe()


# In[29]:


input2.age.plot.hist()
plt.show()


# ### Balance Variable 

# In[30]:


input2.balance.describe()


# In[31]:


plt.figure(figsize=[8,2])
sns.boxplot(input2.balance)
plt.show()


# In[32]:


input2.balance.quantile([0.5,0.7,0.9,0.95,0.99])


# In[33]:


input2[input2.balance>15000].describe()


# #### Instead of looking at mean, we could look at quantiles/meadians/percentiles instead.

# ## Standardize Variable

# #### Duration variable

# In[34]:


input2.duration.head(10)


# In[35]:


input2.duration.describe()


# In[36]:


#Converting Sec in mintues by dividing by 60 if "sec" or "min" is available in duration column
input2.duration = input2.duration.apply(lambda x: float(x.split()[0])/60 if x.find("sec")>0 else float(x.split()[0]))


# In[37]:


input2.duration.describe()


# In[38]:


input2.dtypes


# ## Univariate analysis - categorical features
# #### Marital

# In[39]:


input2.marital.value_counts()


# In[40]:


input2.marital.value_counts(normalize = True)


# In[41]:


input2.marital.value_counts(normalize = True).plot.bar()
plt.show()


# In[42]:


#The two job categories are the least and the most contacted by the bank, respectively?
input2.Job.value_counts(normalize= True).plot.barh()

plt.show()


# #### Education Variable

# In[43]:


input2.Education.value_counts(normalize = True)


# In[44]:


input2.Education.value_counts(normalize = True).plot.pie()
plt.show()


# #### Poutcome variable

# In[45]:


input2.poutcome.value_counts(normalize = True).plot.bar()
plt.show()


# In[46]:


input2[~(input2.poutcome=="unknown")].poutcome.value_counts(normalize = True).plot.bar()
plt.show()


# #### response - the target variable

# In[47]:


input2.response.value_counts(normalize = True)


# In[48]:


input2.response.value_counts(normalize = True).plot.pie()
plt.show()


# # Bivariate analysis

# #### Numerical - numerical

# In[49]:


plt.scatter(input2.salary,input2.balance)
plt.show()


# In[50]:


input2.plot.scatter(x="age", y="balance")
plt.show()


# In[51]:


sns.pairplot(data=input2, vars=["salary","balance","age"])
plt.show()


# ### Quantify using correlation values

# In[52]:


input2[["age","salary","balance"]].corr()


# ### correlation heatmap

# In[53]:


sns.heatmap(input2[["age","salary","balance"]].corr(), annot = True, cmap="Reds")
plt.show()


# ### Categorical - numerical

# In[54]:


input2.groupby("response")["salary"].mean()


# In[55]:


input2.groupby("response")["salary"].median()


# In[56]:


sns.boxplot(data=input2, x = "response", y= "salary")
plt.show()


# ### Response vs Balance
#  - We know that the balance is highly skewed -  has very high value

# In[57]:


sns.boxplot(data=input2, x="response", y="balance")
plt.show()


# In[58]:


input2.groupby('response')['balance'].mean()


# In[59]:


input2.groupby('response')['balance'].median()


# ### What is 75th percentile

# In[60]:


def p75(x):
    return np.quantile(x, 0.75)


# In[61]:


input2.groupby('response')['balance'].aggregate(["mean","median",p75])


# In[62]:


input2.groupby('response')['balance'].aggregate(["mean","median"]).plot.bar()
plt.show()


# #### Which of the following education levels shows the highest mean and the lowest median value for salary?

# In[63]:


input2.groupby('Education')['salary'].mean()


# In[64]:


input2.groupby('Education')['salary'].median()


# ### Categorical to categorical

# In[65]:


#Creating a new variable "response_flag". If the response is YES I'll make is 1 and vice-versa. 
input2['response_flag'] = np.where(input2.response=='yes',1,0)


# In[66]:


input2.response_flag.value_counts()


# In[67]:


input2.response.value_counts()


# In[68]:


input2.response.value_counts(normalize = True)


# In[69]:


input2.response_flag.mean()


# ### Education vs Response rate

# In[71]:


#In the bank marketing campaign dataset, which of the following 
#education categories has the highest percentage of positive responses?
input2.groupby(['Education'])['response_flag'].mean()


# ### Marital vs Response Rate

# In[75]:


input2.groupby(['marital'])['response_flag'].mean()


# In[74]:


input2.groupby(['marital'])['response_flag'].mean().plot.barh()
plt.show()


# In[76]:


input2.groupby(['loan'])['response_flag'].mean().plot.bar()
plt.show()


# In[78]:


input2.groupby(['housing'])['response_flag'].mean().plot.bar()
plt.show()


# ### Age vs Response

# In[83]:


sns.boxplot(data=input2, x="response", y="age")
plt.show()


# ### Making buckets from the age column

# In[84]:


get_ipython().run_line_magic('pinfo', 'pd.cut')


# In[87]:


pd.cut(input2.age[:5], [1,30,40,50,60,9999], labels = ["<30","30-40","40-50","50-60","60+"])


# In[88]:


input2.age.head()


# In[89]:


input2['age_group'] = pd.cut(input2.age, [1,30,40,50,60,9999], labels = ["<30","30-40","40-50","50-60","60+"])


# In[90]:


input2.age_group.value_counts(normalize=True)


# In[94]:


plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
input2.age_group.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
input2.groupby(["age_group"])['response_flag'].mean().plot.bar()
plt.show()


# In[ ]:


input2.groupby(['housing'])['response_flag'].mean().plot.barh()
plt.show()


# ### More than 2 variables

# In[99]:


res = pd.pivot_table(data=input2, index='Education', columns='marital', values='response_flag')
res


# In[100]:


sns.heatmap(res, annot=True,)


# In[101]:


sns.heatmap(res, annot=True, cmap='RdYlGn')


# In[104]:


sns.heatmap(res, annot=True, cmap='RdYlGn', center=0.117)
plt.show()


# ### Job vs Marital vs response

# In[105]:


res = pd.pivot_table(data=input2, index='Job', columns='marital', values='response_flag')
sns.heatmap(res, annot=True, cmap='RdYlGn', center=0.117)
plt.show()


# ### Education vs Poutcome vs response

# In[106]:


res = pd.pivot_table(data=input2, index='Education', columns='poutcome', values='response_flag')
sns.heatmap(res, annot=True, cmap='RdYlGn', center=0.117)
plt.show()


# In[108]:


input2[input2.pdays>0].response_flag.mean()


# In[110]:


res = pd.pivot_table(data=input2, index='Education', columns='poutcome', values='response_flag')
sns.heatmap(res, annot=True, cmap='RdYlGn', center=0.2308)
plt.show()


# In[113]:


res = pd.pivot_table(data=input2, index="Job", columns = "marital", values = "response_flag")
sns.heatmap(res, annot=True, cmap="RdYlGn", center = 0.117)
plt.show()


# In[114]:


res1 = pd.pivot_table(data=input2, index="Education", columns = "marital", values = "response_flag")
res1
sns.heatmap(res1, annot=True, cmap="RdYlGn", center = 0.117)
plt.show()


# # Thanks you

# ## My GitHub Repo: https://github.com/bunty328/Data_Science_Projects

# In[ ]:




