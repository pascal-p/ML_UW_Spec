## C02w02: Multiple Regression

[Machine Learning(ML) Specialization](https://www.coursera.org/specializations/machine-learning)
  - [ML Regression Course](https://www.coursera.org/learn/ml-regression/home/welcome)

### Exploring different multiple regression models for house price prediction
In this notebook we will use data on house sales in King County to predict prices using multiple regression. The first assignment will be about exploring multiple regression in particular exploring the impact of adding features to a regression and measuring error. In the second assignment you will implement a gradient descent algorithm. In this assignment we will:

  - Use SFrames to do some feature engineering
  - Use built-in Turi Create (or otherwise) functions to compute the regression weights (coefficients)
  - Given the regression weights, predictors and outcome write a function to compute the Residual Sum of Squares
  - Look at coefficients and interpret their meanings
  - Evaluate multiple models via RSS

### Implementing gradient descent for multiple regression

Estimating Multiple Regression Coefficients (Gradient Descent)

In the first notebook we explored multiple regression using Turi Create. Now we will use SFrames along with numpy to solve for the regression weights with gradient descent.

In this notebook we will cover estimating multiple regression weights via gradient descent. We will:

  - Add a constant column of 1's to a SFrame (or otherwise) to account for the intercept
  - Convert an SFrame into a numpy array
  - Write a predict_output() function using numpy
  - Write a numpy function to compute the derivative of the regression weights with respect to a single feature
  - Write gradient descent function to compute the regression weights given an initial weight vector, step size and tolerance.
  - Use the gradient descent function to estimate regression weights for multiple features


<hr />

Addition;
  - Complete port to [Julia](https://www.julialang.org/)

Outcome:
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w02/C02w02_nb_pa1.ipynb)
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w02/C02w02_nb_pa2.ipynb)
  - [Julia Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w02/C02w02_nb_pa1.jl)
  - [Julia Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w02/C02w02_nb_pa2.jl)

<hr />
<p><sub><em>Feb. 2021 Corto Inc</sub></em></p>
