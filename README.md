# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries
    Import required libraries: sklearn.datasets, sklearn.model_selection, sklearn.linear_model,      sklearn.metrics.
2.Load Dataset
    Load the Iris dataset using load_iris() from sklearn.datasets.
3.Split Dataset
     Split the dataset into training and testing sets using train_test_split().
4.Initialize Classifier
     Create an instance of SGDClassifier().
5.Train Model
      Fit the classifier on the training data using .fit(X_train, y_train).
6.Make Predictions
      Use .predict(X_test) to predict the species for test data.
7.Evaluate Model
      Measure accuracy using accuracy_score(y_test, y_pred).
8.Output Results
      Print the accuracy and compare predicted vs actual species.

## Program:

Program to implement the prediction of iris species using SGD Classifier.
Developed by: KARTHICK KISHORE T
RegisterNumber: 212223220042  
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:
<img width="485" height="252" alt="7 1" src="https://github.com/user-attachments/assets/9dce49cc-464d-4865-ac64-066ddf6eca5c" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
