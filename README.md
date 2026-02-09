# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries 
2. Load and Prepare the Dataset
3. Train the Decision Tree Model
4. Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.Keerthana
RegisterNumber:  25004216
*/


import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])

```

## Output:
<img width="1246" height="192" alt="Screenshot 2026-02-09 085002" src="https://github.com/user-attachments/assets/8e281a5f-4513-4bcd-be5c-91cd7d17cd95" />
<img width="535" height="364" alt="Screenshot 2026-02-09 085103" src="https://github.com/user-attachments/assets/f89e6dbd-8048-4cfa-90ec-19063b45ccbc" />
<img width="311" height="231" alt="Screenshot 2026-02-09 085129" src="https://github.com/user-attachments/assets/a2cb7d3d-d9da-4e32-9663-48c2a7e89c84" />
<img width="287" height="74" alt="Screenshot 2026-02-09 085206" src="https://github.com/user-attachments/assets/93dd6d9a-6785-422c-9bf4-5a8630a2509f" />
<img width="1286" height="222" alt="Screenshot 2026-02-09 085241" src="https://github.com/user-attachments/assets/5bdead84-601a-455e-9795-09868c679aac" />
<img width="126" height="24" alt="Screenshot 2026-02-09 085343" src="https://github.com/user-attachments/assets/4771e8ae-9342-447b-adab-eac6435f92d2" />
<img width="1249" height="112" alt="Screenshot 2026-02-09 085418" src="https://github.com/user-attachments/assets/7ef87cbe-f3cc-4c61-b161-c64be1d55fd7" />









## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
