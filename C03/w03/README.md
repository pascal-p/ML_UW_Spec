## C03w03: Decision Trees

[Machine Learning(ML) Specialization](https://www.coursera.org/specializations/machine-learning)
  - [ML Classification Course](https://www.coursera.org/learn/ml-classification/home/welcome)

### Identifying safe loans with decision trees

The LendingClub is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors. In this notebook, we will build a classification model to predict whether or not a loan provided by LendingClub is likely to default.

In this notebook we will use data from the LendingClub to predict whether a loan will be paid off in full or the loan will be charged off and possibly go into default.
  - Use SFrames to do some feature engineering.
  - Train a decision-tree on the LendingClub dataset.
  - Visualize the tree.
  - Predict whether a loan will default along with prediction probabilities (on a validation set).
  - Train a complex tree model and compare it to simple tree model.


### Implementing binary decision trees

The goal of this notebook is to implement our own binary decision tree classifier.
  - Use SFrames to do some feature engineering.
  - Transform categorical variables into binary variables.
  - Write a function to compute the number of misclassified examples in an intermediate node.
  - Write a function to find the best feature to split on.
  - Build a binary decision tree from scratch.
  - Make predictions using the decision tree.
  - Evaluate the accuracy of the decision tree.
  - Visualize the decision at the root node.

Important Note: In this assignment, we will focus on building decision trees where the data contain only binary (0 or 1) features.
This allows us to avoid dealing with:

 - Multiple intermediate nodes in a split
 - The thresholding issues of real-valued features.

<hr />

Addition;
  - complete port to [Julia](https://www.julialang.org/)

Outcome:
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C03/w03/C03w03_nb_pa1.ipynb)
  - [Jupyter Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C03/w03/C03w03_nb_pa2.ipynb)
  - [Pluto Notebook/`Julia`/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C03/w03/C03w03_nb_pa1.jl)
  - [Pluto Notebook/`Julia`/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C03/w03/C03w03_nb_pa2.jl)


<hr />
<p><sub><em>Feb.-Mar. 2021 Corto Inc</sub></em></p>
