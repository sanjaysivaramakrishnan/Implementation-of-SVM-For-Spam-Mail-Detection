# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\spam.csv", encoding="Windows-1252")

# View data info
data.info()

# Extract features and labels
x = data['v2'].values  # Text messages
y = data['v1'].values  # Labels (ham/spam)

# Check shapes
print("x shape:", x.shape)
print("y shape:", y.shape)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Convert text to feature vectors
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train the model
svc = SVC()
svc.fit(x_train, y_train)

# Make predictions
y_pred = svc.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


*/
```

## Output:
![image](https://github.com/user-attachments/assets/7cd0515a-07ef-4e85-986e-c4cba8dce987)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
