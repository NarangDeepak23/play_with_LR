import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\Py\Colab Work\ML\Linear Regression\new_insurance_data.csv")

data.head()

data.tail()

data.describe()

data.count()

# data=data.dropna()

data.count()



data.info()

data.isnull().sum()

data["region"].mode()[0]

# EDA
col=[]
col=data.columns
col

for i in col:
    if data[i].dtypes=="int64" or data[i].dtypes=="float":
        data[i]=data[i].fillna(data[i].mean())
    else:
        data[i]=data[i].fillna(data[i].mode()[0])
        
    

data.count()

plt.boxplot(data["bmi"])
plt.xlabel("bmi")

for i in col:
    if data[i].dtypes!='object':
        plt.boxplot(data[i])
        plt.xlabel(i)
        plt.show()

q1=data["bmi"].quantile(0.25) 
q3=data["bmi"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["bmi"]>=LL) & (data["bmi"]<=UL)]

q1=data["past_consultations"].quantile(0.25)
q3=data["past_consultations"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["past_consultations"]>=LL) & (data["past_consultations"]<=UL)]

q1=data["Hospital_expenditure"].quantile(0.25)
q3=data["Hospital_expenditure"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["Hospital_expenditure"]>=LL) & (data["Hospital_expenditure"]<=UL)]

q1=data["NUmber_of_past_hospitalizations"].quantile(0.25) 
q3=data["NUmber_of_past_hospitalizations"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["NUmber_of_past_hospitalizations"]>=LL) & (data["NUmber_of_past_hospitalizations"]<=UL)]

q1=data["Anual_Salary"].quantile(0.25)  
q3=data["Anual_Salary"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["Anual_Salary"]>=LL) & (data["Anual_Salary"]<=UL)]

q1=data["charges"].quantile(0.25)  
q3=data["charges"].quantile(0.75)
IQR=q3-q1
UL=q3+IQR*1.5
LL=q1-IQR*1.5
data=data[(data["charges"]>=LL) & (data["charges"]<=UL)]


data.count()

for i in col:
    if data[i].dtypes!='object':
        plt.boxplot(data[i])
        plt.xlabel(i)
        plt.show()

# MODEL BUILDING

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

data.corr()

# VIF should be minimum to have min multi-collinearity
# VIF<=6
col_lst=[]
for i in col:
    if (data[i].dtypes!="object") & (i !="charges") & (i != "NUmber_of_past_hospitalizations"):
        col_lst.append(i)


col_lst

from statsmodels.stats.outliers_influence import variance_inflation_factor

x=data[col_lst]

x

y=data["charges"]

y

for i in range(len(col_lst)):
    print(i)


x.values

vif_data=pd.DataFrame()
vif_data["Features"]=x.columns
vif_data["VIF_VALUE"]= [variance_inflation_factor (x.values,i) for i in range(len(col_lst))]

vif_data

for i in range(len(col_lst)):
    vif_data["VIF"]=[variance_inflation_factor (x.values,i) for i in range(len(col_lst))]

vif_data

data=data.drop(["num_of_steps"], axis=1)

col=list(data.columns)
col_lst=[]
for i in col:
    if (data[i].dtypes!="object") & (i !="charges") & (i != "NUmber_of_past_hospitalizations"):
        col_lst.append(i)
        
x=data[col_lst]
y=data["charges"]
vif_data=pd.DataFrame()
vif_data["Features"]=x.columns
vif_data["VIF"]=[variance_inflation_factor(x.values,i) for i in range(len(col_lst))]
print(vif_data)

data=data.drop(["bmi"], axis=1)

col=list(data.columns)
col_lst=[]
for i in col:
    if (data[i].dtypes!="object") & (i !="charges") & (i != "NUmber_of_past_hospitalizations"):
        col_lst.append(i)
        
x=data[col_lst]
y=data["charges"]
vif_data=pd.DataFrame()
vif_data["Features"]=x.columns
vif_data["VIF"]=[variance_inflation_factor(x.values,i) for i in range(len(col_lst))]
print(vif_data)

data=data.drop(["age"], axis=1)

col=list(data.columns)
col_lst=[]
for i in col:
    if (data[i].dtypes!="object") & (i !="charges") & (i != "NUmber_of_past_hospitalizations"):
        col_lst.append(i)
        
x=data[col_lst]
y=data["charges"]
vif_data=pd.DataFrame()
vif_data["Features"]=x.columns
vif_data["VIF"]=[variance_inflation_factor(x.values,i) for i in range(len(col_lst))]
print(vif_data)

X=data.loc[:,["children","Claim_Amount","past_consultations","Hospital_expenditure","Anual_Salary"]]
Y=data.loc[:,["charges"]]

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8, random_state=1)

L_model=LinearRegression()

L_model.fit(x_train,y_train)

predict_y=L_model.predict(x_test)

err=y_test-predict_y

err

# or we can write in decorative way
err_data=pd.DataFrame()

err_data["Original Test Data"]=y_test
err_data["Predicted Data"]=predict_y
err_data["Error"]=y_test-predict_y

err_data

# Check Accuracy
from sklearn.metrics import *

r2score=r2_score(y_test,predict_y)

r2score

square_error=mean_squared_error(y_test,predict_y,squared=False)
square_error

from sklearn.metrics import mean_absolute_percentage_error

map_error=mean_absolute_percentage_error(y_test,predict_y)
map_error

