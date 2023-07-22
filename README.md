# Employee_Burnout

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from google.colab import drive
drive.mount('/content/drive')

pd.set_option('display.max_columns',None)
burnoutDf=pd.read_csv('/content/drive/MyDrive/employee_burnout.csv')
burnoutDf

burnoutDf["Date of Joining"]=pd.to_datetime(burnoutDf["Date of Joining"])
burnoutDf.shape

burnoutDf.info()

burnoutDf.head()

burnoutDf.columns

burnoutDf.isnull().sum()

burnoutDf.duplicated().sum()

burnoutDf.describe()

for i, col in enumerate(burnoutDf.columns):
  print(f"\n\n{burnoutDf[col].unique()}")
  print(f"\n{burnoutDf[col].value_counts()}\n\n")

burnoutDf=burnoutDf.drop(['Employee ID'],axis=1)
intFloatburnoutDf=burnoutDf.select_dtypes([np.int,np.float])
for i, col in enumerate(intFloatburnoutDf.columns):
  if(intFloatburnoutDf[col].skew()>=0.1):
    print("\n",col,"feature is Positively skewed and value is:",intFloatburnoutDf[col].skew())
  elif(intFloatburnoutDf[col].skew()<=-0.1):
     print("\n",col,"feature is Negtively skewed and value is:",intFloatburnoutDf[col].skew())
  else:
    print("\n",col,"feature is Normally Distributed and value is:",intFloatburnoutDf[col].skew())


burnoutDf['Resource Allocation'].fillna(burnoutDf['Resource Allocation'].mean(),inplace=True)
burnoutDf['Mental Fatigue Score'].fillna(burnoutDf['Mental Fatigue Score'].mean(),inplace=True)
burnoutDf['Burn Rate'].fillna(burnoutDf['Burn Rate'].mean(),inplace=True)
burnoutDf.isna().sum()

burnoutDf.corr()

Corr=burnoutDf.corr()
sns.set(rc={'figure.figsize':(14,12)})
fig=px.imshow(Corr,text_auto=True,aspect="auto")
fig.show()

plt.figure(figsize=(10,8))
sns.countplot(x='Gender',data=burnoutDf,palette="magma")
plt.title("plot distribution of gender")
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x='Company Type',data=burnoutDf,palette="Spectral")
plt.title("plot distribution of Company Type")
plt.show()


plt.figure(figsize=(10,8))
sns.countplot(x='WFH Setup Available',data=burnoutDf,palette="dark:salmon_r")
plt.title("plot distribution of WFh_setup_Avaiable")
plt.show()

burn_st=burnoutDf.loc[:,'Date of Joining':'Burn Rate']
burn_st=burn_st.select_dtypes([int,float])
for i,col in enumerate(burn_st.columns):
  fig=px.histogram(burn_st,x=col,title="plot Distribution of"+col,color_discrete_sequence=['indianred'] )
  fig.update_layout(bargap=0.2)
  fig.show()

fig=px.line(burnoutDf,y="Burn Rate",color="Designation",title="Burn rate on the basis of Designation",color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.1)
fig.show()


fig=px.line(burnoutDf,y="Burn Rate",color="Gender",title="Burn rate on the basis of Gender",color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.2)
fig.show()

fig=px.line(burnoutDf,y="Mental Fatigue Score",color="Designation",title="Mental fatigue vs designation",color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_layout(bargap=0.2)
fig.show()

sns.relplot(
    data=burnoutDf,x="Designation",y="Mental Fatigue Score",col="Company Type",
    hue="Company Type",size="Burn Rate",style="Gender",
    palette=["g","r"],sizes=(50,200)
)

from sklearn import preprocessing
Label_encode=preprocessing.LabelEncoder()
burnoutDf['GenderLabel'] = Label_encode.fit_transform(burnoutDf[ "Gender"].values)
burnoutDf['Company_TypeLabel'] = Label_encode.fit_transform(burnoutDf['Company Type'].values)
burnoutDf['WFH_Setup_AvailableLabel']=Label_encode.fit_transform(burnoutDf['WFH Setup Available'].values)

gn=burnoutDf.groupby('Gender')
gn=gn['GenderLabel']
gn.first()

ct = burnoutDf.groupby('Company Type')
ct = ct['Company_TypeLabel']
ct.first()

wsa=burnoutDf.groupby('WFH Setup Available')
wsa=wsa['WFH_Setup_AvailableLabel']
wsa.first()

burnoutDf.tail(10)

Columns=['Designation','Resource Allocation','Mental Fatigue Score','GenderLabel','Company_TypeLabel','WFH_Setup_AvailableLabel']
x=burnoutDf[Columns]
y=burnoutDf['Burn Rate']
print(x)

print(y)

from sklearn.decomposition import PCA
pca=PCA(0.95)
x_pca=pca.fit_transform(x)
print("pca shape of x is:",x_pca.shape,"and original shape is:",x.shape)
print("% of importance of selected features is",pca.explained_variance_ratio_)
print("The number of features selected through pca is:",pca.n_components_)

from sklearn.model_selection import train_test_split
x_train_pca,x_test,y_train,y_test=train_test_split(x_pca,y,test_size=0.25,random_state=10)
print(x_train_pca.shape,x_test.shape,y_train.shape,y_test.shape)

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(x_train_pca, y_train)

train_pred_rf=rf_model.predict(x_train_pca)
train_r2=r2_score(y_train, train_pred_rf)
test_pred_rf=rf_model.predict(x_test)
test_r2=r2_score(y_test, test_pred_rf)
print("Accuracy score of tarin data: "+str(round(100*train_r2, 4))+" %")
print("Accuracy score of test data: "+str(round(100*test_r2, 4))+" %")

from sklearn.ensemble import AdaBoostRegressor
abr_model=AdaBoostRegressor()
abr_model.fit(x_train_pca,y_train)

train_pred_adboost=abr_model.predict(x_train_pca)
train_r2=r2_score(y_train, train_pred_adboost)
test_pred_adaboost=abr_model.predict(x_test)
test_r2 = r2_score(y_test, test_pred_adaboost)
print("Accuracy score of tarin data: "+str(round(100*train_r2, 4))+" %")
print("Accuracy score of test data: "+str(round (100*test_r2, 4))+" %")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_pca,y,test_size=0.25,random_state=10)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)

train_pred_rf=rf_model.predict(x_train)
train_r2=r2_score(y_train, train_pred_rf)
test_pred_rf=rf_model.predict(x_test)
test_r2=r2_score(y_test, test_pred_rf)
print("Accuracy score of train data after random forest regression: "+str(round(100*train_r2, 4))+" %")
print("Accuracy score of test data after random forest regression: "+str(round(100*test_r2, 4))+" %")

from sklearn.ensemble import AdaBoostRegressor
abr_model=AdaBoostRegressor()
abr_model.fit(x_train,y_train)
