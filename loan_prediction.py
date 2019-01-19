
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


Total_train=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
test=pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
Total_train.head(3)
#Total_train.shape
#Total_test.shape


# In[ ]:


train_original=Total_train.copy()
test_original=test.copy()
#print(Total_train.isnull().sum())
Total_train.columns


# In[67]:


Total_train.Gender.fillna(Total_train.Gender.dropna().max(),inplace=True)
Total_train.Married.fillna(Total_train.Married.dropna().max(),inplace=True)
Total_train.Credit_History.fillna(Total_train.Credit_History.max(),inplace=True)
Total_train.LoanAmount.fillna(Total_train.LoanAmount.mean(),inplace=True)
Total_train.Loan_Amount_Term.fillna(Total_train.Loan_Amount_Term.mean(),inplace=True)
Total_train.Self_Employed.fillna(Total_train.Self_Employed.dropna().max(),inplace=True)
Total_train.Dependents.fillna(0,inplace=True)


Total_train.Gender.value_counts()
gender_cat=pd.get_dummies(Total_train.Gender,prefix='gender').gender_Female

Total_train.Married.value_counts()
Married_cat=pd.get_dummies(Total_train.Married,prefix='marriage').marriage_Yes

Total_train.Education.value_counts()
graduate_cat=pd.get_dummies(Total_train.Education,prefix='education').education_Graduate

Total_train.Self_Employed.value_counts()
self_emp_cat=pd.get_dummies(Total_train.Self_Employed,prefix='employed').employed_Yes

loan_status=pd.get_dummies(Total_train.Loan_Status,prefix='status').status_Y

property_cat=pd.get_dummies(Total_train.Property_Area,prefix='property')

Total_train.shape
#print(Total_train.isnull().sum())


# In[68]:


trainnew=pd.concat([Total_train,gender_cat,Married_cat,graduate_cat,self_emp_cat,loan_status,property_cat],axis=1)
trainnew.head()


# In[69]:


trainnew.columns


# In[70]:


features=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','gender_Female','marriage_Yes',
          'education_Graduate','employed_Yes','property_Rural','property_Semiurban','property_Urban']
x=trainnew[features]
y=trainnew['status_Y']
#y_train


# In[71]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.01,random_state=42)


# In[72]:


#y_train.shape


# In[73]:


randForest = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForest.fit(x_train,y_train)
y_pred=randForest.predict(x_test)
randForestScore=accuracy_score(y_test,y_pred)
print(randForestScore)


# In[74]:


randForestnew = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForestnew.fit(x,y)


# In[76]:


test.shape


# In[79]:


test.head()
#print(test.isnull().sum())


# In[88]:


test.Gender.fillna(test.Gender.dropna().max(),inplace=True)
test.Married.fillna(test.Married.dropna().max(),inplace=True)
test.Credit_History.fillna(test.Credit_History.max(),inplace=True)
test.LoanAmount.fillna(test.LoanAmount.mean(),inplace=True)
test.Loan_Amount_Term.fillna(test.Loan_Amount_Term.mean(),inplace=True)
test.Self_Employed.fillna(test.Self_Employed.dropna().max(),inplace=True)
test.Dependents.fillna(0,inplace=True)


# In[89]:


gender_cat=pd.get_dummies(test.Gender,prefix='gender').gender_Female
Married_cat=pd.get_dummies(test.Married,prefix='marriage').marriage_Yes
graduate_cat=pd.get_dummies(test.Education,prefix='education').education_Graduate
self_emp_cat=pd.get_dummies(test.Self_Employed,prefix='employed').employed_Yes
property_cat=pd.get_dummies(test.Property_Area,prefix='property')

test.shape
#print(test.isnull().sum())


# In[91]:


testnew=pd.concat([test,gender_cat,Married_cat,graduate_cat,self_emp_cat,property_cat],axis=1)
testnew.columns


# In[97]:


X_test=testnew[features]


# In[95]:


X_test.head()


# In[98]:


Y_pred=randForestnew.predict(X_test)
randForestFormat=["Y" if i==1 else "N" for i in Y_pred]
pd.DataFrame({'Loan_ID':testnew.Loan_ID,'Loan_Status':randForestFormat}).to_csv('random_forest_submission.csv',index=False)
result=pd.read_csv('random_forest_submission.csv')
result.head()

