#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"D:\Py\Colab Work\ML\Logistic Regression\bank-additional-full.csv", sep=";") #here seperator is npt identified

# here target variable is binary that is having two categories like patient is having diabetes or not


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


col=list(df.columns)
col


# In[8]:


# for i in col:
#     if (df[i].dtypes=="int32" or df[i].dtypes=="float"):
#         df[i]=df[i].fillna(df[i].mean())
#     else:
#         df[i]=df[i].fillna(df[i].mode()[0]) 


# In[9]:


# df.isnull().sum()


# In[10]:


import seaborn as sns
for i in col:
    if df[i].dtypes!="object":
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.ylabel("count")
        plt.show()
        


# In[11]:


df["marital"].value_counts()


# In[12]:


df["loan"].value_counts()


# In[13]:


df["marital"].value_counts().tolist()


# In[14]:


cols=['age','duration','campaign','cons.conf.idx']


# In[15]:


q1=df['age'].quantile(0.25)
q3=df['age'].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
df=df[(df["age"]<=UL) & (df["age"]>=LL)]

q1=df['duration'].quantile(0.25)
q3=df['duration'].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
df=df[(df['duration']<=UL) & (df['duration']>=LL)]

q1=df['campaign'].quantile(0.25)
q3=df['campaign'].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
df=df[(df['campaign']<=UL) & (df['campaign']>=LL)]

q1=df['cons.conf.idx'].quantile(0.25)
q3=df['cons.conf.idx'].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
df=df[(df['cons.conf.idx']<=UL) & (df['cons.conf.idx']>=LL)]


# In[16]:


for i in col:
    if df[i].dtypes!="object":
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.ylabel("count")
        plt.show()


# In[17]:


df.isnull().sum()


# In[18]:


df["y"]=np.where(df["y"]=="yes",1,0)


# In[19]:


df["y"].dtypes


# In[20]:


df["y"]=df["y"].astype(str)


# In[21]:


df["y"].dtypes


# In[22]:


col=list(df.columns)
for i in col:
    if (df[i].dtypes=="int32" or df[i].dtypes=="float"):
        df[i]=df[i].fillna(df[i].mean())
    else:
        df[i]=df[i].fillna(df[i].mode()[0]) 


# In[23]:


df.isnull().sum()


# In[24]:


# FEATURE SELECTION
# binning the data
bins=[0,50,100]
df["age"]=pd.cut(df["age"],bins)
df["age"]=df["age"].astype(str)


# In[25]:


df["age"].value_counts()/len(df["age"])


# In[26]:


df


# In[27]:


# IV analysis (Information Value)
"""
done to check the predictive power of feature
* by binning the values 
* and by finding th weight of confidence
% of obs under events 
% of events under non-events
take log of this
Target has dichotomous
has two events : events and non-events
"""


# In[28]:


def cal_woe(dataset, feature, target):
    lst=[]
    for i in range(dataset[feature].nunique()):
        val=list(dataset[feature].unique())[i]
        lst.append({
            "value":val,
            "All":dataset[(dataset[feature]==val)].count()[feature],
            "Good":dataset[(dataset[feature]==val) & (dataset[target]==1)].count()[feature],
            "Bad":dataset[(dataset[feature]==val)& (dataset[target]==0)].count()[feature]
        })
    dset=pd.DataFrame(lst)
    dset["Dist_Good"]=dset["Good"]/dset["Good"].sum()
    dset["Dist_Bad"]=dset["Bad"]/dset["Bad"].sum()
    dset["WOE"]=np.log(dset["Dist_Good"]/dset["Dist_Bad"])
    dset=dset.replace({"WOE":{np.inf:0,-np.inf:0}})
    dset["IV"]=(dset["Dist_Good"]-dset["Dist_Bad"])*dset["WOE"]
    iv=dset["IV"].sum()
    dset=dset.sort_values(by="WOE")
    return dset, iv


# In[29]:


col_lst=list(df.columns)


# In[30]:


df_new=pd.DataFrame(columns={"Feature","IV Score"})
df["y"]=df["y"].astype(int)
for i in col_lst:
    if i=="y":
        continue
    elif df[i].dtypes=='object':
        df1,iv=cal_woe(df,i,"y")
        df_new=df_new.append({"Feature":i, "IV Score":iv}, ignore_index=True)
df_new
        


# In[31]:


df=df.drop(columns={"age","marital","education","housing","loan","day_of_week"})


# In[32]:


df


# # ONE HOT ENCODING
# # it creats a seprate column for each feature present in each column
# # for k features it creates k-1 features
# 

# In[33]:


cols=[]
for i in (df.columns):
    if (df[i].dtypes=="object") & (i!="y"):
        cols.append(i)


# In[34]:


cols


# In[35]:


df2=pd.get_dummies(df[cols],drop_first=True)


# In[36]:


for i in (df2.columns):
    df2[i]=df2[i].astype(int)


# In[37]:


df2


# In[38]:


df_new=pd.concat([df,df2],axis=1)


# In[39]:


df_new


# In[40]:


df_new.drop(columns=cols,axis=1, inplace=True)


# In[41]:


df_new


# #Label Encoding 
# here priority of features in a column is made on the basis of alphabetical order 
# and 0 is at least priority

# In[42]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[43]:


for i in cols:
    df[i]=LE.fit_transform(df[i])


# In[44]:


df.head()


# In[45]:


df.shape


# In[46]:


# VIF CALCULATION
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[47]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]


# In[48]:


vif_data


# In[49]:


col_lst


# In[50]:


df=df.drop(columns=["nr.employed"], axis=1)


# In[51]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]


# In[52]:


vif_data


# In[53]:


df=df.drop(columns=["cons.price.idx"],axis=1)


# In[54]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif_data


# In[56]:


df=df.drop(columns=["pdays"],axis=1)


# In[57]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif_data


# In[58]:


df=df.drop(columns=["euribor3m"],axis=1)


# In[59]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif_data


# In[60]:


df=df.drop(columns=["cons.conf.idx"],axis=1)


# In[61]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif_data


# In[62]:


df=df.drop(columns=["poutcome"],axis=1)


# In[63]:


col_lst=[]
for i in (df.columns):
    if (df[i].dtypes!="object")&(i!="y"):
        col_lst.append(i)

X=df[col_lst]
vif_data=pd.DataFrame()
vif_data["Features"]=X.columns
vif_data["VIF"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif_data


# In[64]:


# MODEL BUILDING
from sklearn.model_selection import train_test_split


# In[65]:


from sklearn.linear_model import LogisticRegression


# In[66]:


df


# In[69]:


X=df.iloc[:,0:8]


# In[70]:


X


# In[72]:


Y=df.iloc[:,-1]


# In[73]:


Y


# In[74]:


lr=LogisticRegression()


# In[75]:


x_train,x_test,y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[76]:


lr.fit(x_train,y_train)


# In[77]:


y_predict=lr.predict(x_test)


# In[78]:


Error=pd.DataFrame()


# In[81]:


Error["Y_org"]=Y_test
Error["Y_predicted"]=y_predict
Error["Error"]=Y_test-y_predict
Error


# In[83]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[84]:


accuracy_score(Y_test,y_predict)


# In[85]:


print(confusion_matrix(Y_test,y_predict))


# In[87]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,y_predict))


# In[ ]:




