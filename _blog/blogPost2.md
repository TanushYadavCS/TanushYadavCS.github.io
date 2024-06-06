---
title: 'Simple Linear Regression'
date: 2024-06-05
permalink: /blog/2024/06/blog-post-2/
tags:
  - regression
  - linear regression
  - machine learning
---

## What is Regression?
> Regression is the most basic yet most fundamental concept that one needs to be familiar with while studying ML.


Simply put, regression is a method used to predict the relationship between **dependent *(target / outcome)* variables** and **independent *(predictors / features)* variables**.\\
\\
Now, many forms of regression are used in machine learning but in this post, we will focus on **implementing simple linear regression**.
## Coming to Simple Linear Regression
Firstly, linear regression is a type of regression which focuses on trying to establish a linear relationship between the dependent and independent variable.\\
\\
Simply put, linear regression tries to find a straight line that has the least distance from each data point when the dependent and independent variables are plotted on a graph.\\
<img src="https://i.postimg.cc/Pxs4vHTP/linear-Regression.png">

>If linear regression is implemented to predict the value of a dependent variable based on a single dependent variable, it is termed as **simple linear regression**.

## The Math 
- ### Regression Line
Since the regression line is a straight line, its formula will be
> $$ y = mX + c $$
> 
> **where:**
> 
> -  $$y$$  denotes the dependent variable, target
> -  $$m$$  represents the slope of the line
> -  $$X$$  signifies the independent variable, feature
> -  $$c$$  is the y-intercept
- ### Forward Pass
> The forward pass function is used to calculate the predicted values of the model.

  The forward pass formula is
  > $$ \hat{y} = mX + c $$
  > 
  > **where:**
  > 
  > -  $$\hat{y}$$  denotes the dependent variable, target
  > -  $$m$$  represents the slope of the line
  > -  $$X$$  signifies the independent variable, feature
  > -  $$c$$  is the y-intercept
- ### Cost Function
> The cost function is used to determine the performance of the model. It quantifies the error between the predicted and actual values.
  
  In this post, we will be using the mean squared error metric for our cost function. The formula is
  > $$ \text{MSE} = \frac{\sum (y - \hat{y})^2}{n} $$
  > 
  > **where:**
  > 
  > - $$MSE$$ denotes the Mean Squared Error
  > - $$y$$ represents the actual values
  > - $$\hat{y}$$ signifies the predicted values
  > - $$n$$ is the number of observations
- ### Backward Pass
> The backward pass function is used to determine how a small change in the parameters will impact the function. 

  This is done so by calculating the gradient of the cost function with respect to $$m$$ and $$c$$.
  > $$ \frac{\partial \text{MSE}}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} X_i (y_i - \hat{y}_i) $$
  > 
  > $$ \frac{\partial \text{MSE}}{\partial c} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) $$
  > 
  > **where:**
  > 
  > - $$ \frac{\partial \text{MSE}}{\partial m} $$ denotes the partial derivative of MSE with respect to the slope $$ m $$
  > - $$ \frac{\partial \text{MSE}}{\partial c} $$ denotes the partial derivative of MSE with respect to the intercept $$ c $$
  > - $$ n $$ is the number of observations
  > - $$ X_i $$ represents the input feature for the $$ i $$-th observation
  > - $$ y_i $$ represents the actual target value for the $$ i $$-th observation
  > - $$ \hat{y}_i $$ signifies the predicted value for the $$ i $$-th observation
- ### Weight Update
> We use this function to update the values of m and c using the computed gradients.

  This step is generally achieved using gradient descent.
  > $$ m = m - \alpha \frac{\partial \text{MSE}}{\partial m} $$
  > 
  > $$ c = c - \alpha \frac{\partial \text{MSE}}{\partial c} $$
  > 
  > **where:**
  > 
  > - $$ m $$ is the slope (weight)
  > - $$ c $$ is the intercept (bias)
  > - $$ \alpha $$ is the learning rate
  > - $$ \frac{\partial \text{MSE}}{\partial m} $$ is the gradient of the MSE with respect to the slope
  > - $$ \frac{\partial \text{MSE}}{\partial c} $$ is the gradient of the MSE with respect to the intercept

## The Implementation
Now, we will implement what we have studied so far. 
> One doesn't usually define their own regression models while working with datasets. But, for the sake of learning, we will implement our own linear regression class in Python.

1. ### Import necessary libraries
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.axes as ax
  from matplotlib.animation import FuncAnimation
  plt.style.use('dark_background') # Use dark theme for your graphs (Optional)
  ```
2. ### Import dataset
  ```python
  ds = "./datasets/linearRegression.csv" # URL of your dataset 
  data = pd.read_csv(ds)
  data
  ```
3. ### Split dataset
  ```python
  data = data.dropna()
  X_train = np.array(data.x[0:500]).reshape(500,1)
  y_train = np.array(data.y[0:500]).reshape(500,1)
  X_test = np.array(data.x[500:700]).reshape(199, 1)
  y_test = np.array(data.y[500:700]).reshape(199, 1)
  ```
4. ### Define linear regression model
  ```python
  class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def forward_propagation(self, X_train):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, X_train)+c
        return predictions

    def cost_function(self, predictions, y_train):
        cost = np.mean((y_train-predictions) ** 2)
        return cost

    def backward_propagation(self, X_train, y_train, predictions):
        derivatives = {}
        df = (predictions-y_train)
        dm = 2 * np.mean(np.multiply(X_train, df))
        dc = 2*np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives

    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - \
            learning_rate*derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - \
            learning_rate*derivatives['dc']

    def train(self, X_train, y_train, learning_rate, iters):
        self.parameters['m'] = np.random.uniform(0, 1)*-1
        self.parameters['c'] = np.random.uniform(0, 1)*-1
        self.loss = []
        fig, ax = plt.subplots()
        x_vals = np.linspace(min(X_train), max(X_train), 100)
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals +
                       self.parameters['c'], color='red', label='Regression Line')
        ax.scatter(X_train, y_train, marker='o',
                   color='#28fc03', label='Training Data')
        ax.set_ylim(0, max(y_train)+1)

        def update(frame):
            predictions = self.forward_propagation(X_train)
            cost = self.cost_function(predictions, y_train)
            derivatives = self.backward_propagation(
                X_train, y_train, predictions)
            self.update_parameters(derivatives, learning_rate)
            line.set_ydata(self.parameters['m']*x_vals+self.parameters['c'])
            self.loss.append(cost)
            print("Iteration = {}, Loss={}".format(frame+1, cost))
            return line,
        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
        ani.save('./outputs/linearRegression.gif', writer='ffmpeg', dpi=200, fps=20)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.savefig('./outputs/linearRegression.png', dpi=200)
        plt.show()
        return self.parameters, self.loss
    ```
5. ### Train model and give final prediction
    ```python
    linear_reg = LinearRegression()
    parameters, loss = linear_reg.train(X_train, y_train, 0.0001, 20)
    ```
## Output
After running this program, you will get output similar to this
\\
\\
<img src="https://i.postimg.cc/MKw6GBdh/output.png">
<img src="https://i.postimg.cc/hvhk5Srb/linear-Regression.png" style="width=25%;">
<img src="https://i.postimg.cc/rsbB6vZ6/linear-Regression.gif" style="width=25%;">

# Conclusion
Simple linear regression works by finding the best-fitting line through the data points to predict the relationship between the independent and dependent variables.

 It minimizes the sum of the squared differences between the observed and predicted values, aiming to capture the linear relationship between the variables.
