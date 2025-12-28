import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# data=pd.read_csv('Diabetes_prediction.csv')
# plt.scatter(data['Age'], data['Glucose'], c=data['DiabetesPedigreeFunction'], cmap='viridis')
# plt.show()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def calculate_gradient(theta, X, y):
   m=len(y) # number of instances
   return 1/m * X.T @ (sigmoid(X @ theta) - y)

def gradient_descent(X,y,num_iter=5000,tol=1e-7,learning_rate=0.1):
   x_b=np.c_[np.ones((X.shape[0],1)),X] # add bias term
   theta=np.zeros(x_b.shape[1]) # initialize weights
   for i in range(num_iter):
       gradient=calculate_gradient(theta,x_b,y)
       new_theta=theta - learning_rate * gradient
       if np.linalg.norm(new_theta - theta, ord=1) < tol:
           break
       theta=new_theta    
       
   return new_theta

def predict_proba(X,theta):
    x_b=np.c_[np.ones((X.shape[0],1)),X] # add bias term
    return sigmoid(x_b @ theta)

#return 1 or 0 based on threshold
def predict(X,theta,threshold=0.5):
    proba=predict_proba(X,theta)
    return (proba >= threshold).astype(int)


from sklearn.datasets import load_breast_cancer
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score

X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
theta=gradient_descent(X_train,y_train,learning_rate=0.1)
y_pred_train=predict(X_train,theta,threshold=0.5)
y_pred_test=predict(X_test,theta,threshold=0.5)
print("Train Accuracy:", accuracy_score(y_train,y_pred_train))
print("Test Accuracy:", accuracy_score(y_test,y_pred_test))

