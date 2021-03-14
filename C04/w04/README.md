## C04w04: Mixture Models

[Machine Learning(ML) Specialization](https://www.coursera.org/specializations/machine-learning)
  - [ML Clustering & Retrieval Course](https://www.coursera.org/learn/ml-clustering-and-retrieval/home/welcome)

### Implementing EM for Gaussian mixtures
In this assignment we will

  - implement the EM algorithm for a Gaussian mixture model
  - apply your implementation to cluster images
  - explore clustering results and interpret the output of the EM algorithm

### Clustering text data with Gaussian mixtures

In a previous assignment, we explored K-means clustering for a high-dimensional Wikipedia dataset. We can also model this data with a mixture of Gaussians, though with increasing dimension we run into several important problems associated with using a full covariance matrix for each component.

In this section, we will use an EM implementation to fit a Gaussian mixture model with diagonal covariances to a subset of the Wikipedia dataset.


<hr />

Addition;
  - <em>WIP</em> complete port to [Julia](https://www.julialang.org/)

Outcome:
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C04/w04/C04w04_nb_pa1.ipynb)
  - [Pluto Notebook/PA1`Julia`](https://github.com/pascal-p/ML_UW_Spec/blob/main/C04/w04/C04w04_nb_pa1.jl)
  - [Jupyter Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C04/w04/C04w04_nb_pa2.ipynb)

<hr />
<p><sub><em>Feb.-Mar. 2021 Corto Inc</sub></em></p>
