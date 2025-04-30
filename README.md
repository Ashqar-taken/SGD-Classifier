# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data
2. Split Dataset into Training and Testing Sets
3. Train the Model Using Stochastic Gradient Descent (SGD)
4. Make Predictions and Evaluate Accuracy
5. Generate Confusion Matrix

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ashqar Ahamed S.T
RegisterNumber: 212224240018
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target


print(df.head())


X = df.drop('target', axis=1)
y = df['target']

print("\nTarger Values:\n",y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

print("\nPredicted values:",y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
```

## Data:
![Data](https://github.com/user-attachments/assets/29f4fa86-2427-40ca-bb82-08a954602638)

## Target Values:
![image](https://github.com/user-attachments/assets/a7ce1318-9c6f-4dd7-8301-264969a1210f)

## Prediction and Accuracy:
![image](https://github.com/user-attachments/assets/eebbed88-ffdf-4aa6-8458-f0e2f0e25036)

## Confustion Matrix:
![image](https://github.com/user-attachments/assets/203ac74b-067c-4154-8fdc-eb9cee54177d)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
