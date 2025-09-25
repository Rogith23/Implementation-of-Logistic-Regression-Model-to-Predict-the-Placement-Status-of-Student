# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:
Import the standard libraries such as pandas module to read the corresponding csv file.

Step 2:
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

Step 3:
Import LabelEncoder and encode the corresponding dataset values.

Step 4:
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

Step 5:
Predict the values of array using the variable y_pred.

Step 6:
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

Step 7:
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8:
End the program.u76

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rogith J
RegisterNumber: 212224040280

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Name: Rogith")
print("Register No: 212224040280")
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
 
*/
```

## Output:
HEAD OF THE DATA:

<img width="1167" height="235" alt="Screenshot 2025-09-25 082711" src="https://github.com/user-attachments/assets/41605518-618e-4d0d-83ad-d71b2c843854" />

SUM OD ISNULL DATA:

<img width="361" height="313" alt="Screenshot 2025-09-25 082718" src="https://github.com/user-attachments/assets/05f10946-9bf2-4ef2-8c6f-20e82bb9bd2d" />

SUM OF DUPLICATE DATA:

<img width="280" height="48" alt="Screenshot 2025-09-25 082730" src="https://github.com/user-attachments/assets/731d2d40-54ae-4616-8f9c-8f4cd3b7c228" />

LABEL ENCODER OD DATA:

<img width="1161" height="462" alt="Screenshot 2025-09-25 082738" src="https://github.com/user-attachments/assets/6d00cd12-fc30-436b-a773-f487ce352acb" />

ILOC OF X:

<img width="1048" height="448" alt="Screenshot 2025-09-25 082745" src="https://github.com/user-attachments/assets/57cb716d-73de-43a8-87a0-edf714ffce46" />


<img width="550" height="271" alt="Screenshot 2025-09-25 082752" src="https://github.com/user-attachments/assets/f5a0dd61-97f0-4eec-924f-4ea478b5b4c1" />

PREDICTED VALUES:


<img width="822" height="74" alt="Screenshot 2025-09-25 082801" src="https://github.com/user-attachments/assets/4c818697-119e-452d-b977-905f96e0dee3" />


ACCURACY:


<img width="464" height="38" alt="Screenshot 2025-09-25 082808" src="https://github.com/user-attachments/assets/3ae8d03a-579c-43cb-b338-f7d0f530c87d" />


CONFUSION MATRIX:


<img width="668" height="68" alt="Screenshot 2025-09-25 082815" src="https://github.com/user-attachments/assets/cc27d7a8-29cb-4c43-a591-3c6cc6a74317" />


CLASSIFICATION REPORT:


<img width="1362" height="152" alt="Screenshot 2025-09-25 082824" src="https://github.com/user-attachments/assets/71448e15-6fdd-4553-aec3-1162d05bc5b9" />


PREDICTED LR VALUE:


<img width="572" height="42" alt="Screenshot 2025-09-25 082830" src="https://github.com/user-attachments/assets/c64d5b80-e384-4eec-8b62-112a5428eeef" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
