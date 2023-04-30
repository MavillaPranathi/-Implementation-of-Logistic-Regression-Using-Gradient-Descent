# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules
   from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:M.Pranathi 
RegisterNumber:212222240064  
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:

![image](https://user-images.githubusercontent.com/118343610/235357409-2a18e64b-d1c1-473c-9ebc-e0f26011a251.png)

![image](https://user-images.githubusercontent.com/118343610/235357428-b2c3435b-beb5-4cd0-af59-fa09aaa95643.png)

![image](https://user-images.githubusercontent.com/118343610/235357439-41c7ee2f-4c27-4336-9cf5-6a8b36d25461.png)

![image](https://user-images.githubusercontent.com/118343610/235357453-ae53b40b-6add-43cb-9315-b3942338a303.png)

![image](https://user-images.githubusercontent.com/118343610/235357465-eb567bc9-642b-49dc-b360-65e9434a0db3.png)

![image](https://user-images.githubusercontent.com/118343610/235357498-037efbcc-a282-4501-8a92-27fd7f8dd6d0.png)

![image](https://user-images.githubusercontent.com/118343610/235357513-44ba1a1f-9223-40ee-be66-b1059d749b44.png)

![image](https://user-images.githubusercontent.com/118343610/235357528-9572a474-f487-409f-8e45-c0cbff58f332.png)

![image](https://user-images.githubusercontent.com/118343610/235357543-59b1984e-717a-4ba3-8f57-f2240f106e94.png)

![image](https://user-images.githubusercontent.com/118343610/235357555-a7a2ebcc-7c02-41c3-a9df-d964ad6a54d4.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

