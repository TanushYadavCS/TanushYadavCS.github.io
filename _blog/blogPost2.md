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
> -  y  denotes the dependent variable, target
> -  m  represents the slope of the line
> -  X  signifies the independent variable, feature
> -  c  is the y-intercept