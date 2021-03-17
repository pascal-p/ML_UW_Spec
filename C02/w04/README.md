## C02w04: Ridge Regression

[Machine Learning(ML) Specialization](https://www.coursera.org/specializations/machine-learning)
  - [ML Regression Course](https://www.coursera.org/learn/ml-regression/home/welcome)

### Observing effects of L2 penalty in polynomial regression
In this assignment, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:

  - Use a pre-built implementation of regression to run polynomial regression
  - Use matplotlib to visualize polynomial regressions
  - Use a pre-built implementation of regression to run polynomial regression, this time with L2 penalty
  - Use matplotlib to visualize polynomial regressions under L2 regularization
  - Choose best L2 penalty using cross-validation.
  - Assess the final fit using test data.

We will continue to use the House data from previous assignments.

### Implementing ridge regression via gradient descent
In this assignment, we will implement ridge regression via gradient descent.
We will:
  -  Convert an SFrame into a Numpy array (if applicable)
  - Write a Numpy function to compute the derivative of the regression weights with respect to a single feature
  - Write gradient descent function to compute the regression weights given an initial weight vector, step size, tolerance, and L2 penalty
  - We will continue to use the House data from previous assignments. (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)


<hr />

Addition;
  - Complete port to [Julia](https://www.julialang.org/)

Outcome:
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w04/C02w04_nb_pa1.ipynb)
  - [Julia Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w04/C02w04_nb_pa1.jl)
  - [Jupyter Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w04/C02w04_nb_pa2.ipynb)
  - [Julia Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C02/w04/C02w04_nb_pa2.jl)

<hr />
<p><sub><em>Feb. 2021 Corto Inc</sub></em></p>
