## C04w02: Choosing features and metrics for nearest neighbor search

[Machine Learning(ML) Specialization](https://www.coursera.org/specializations/machine-learning)
  - [ML Clustering & Retrieval Course](https://www.coursera.org/learn/ml-clustering-and-retrieval/home/welcome)

### Choosing features and metrics for nearest neighbor search

When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. 
To find relevant documents we typically
  - Decide on a notion of similarity
  - Find the documents that are most similar

In the assignment we will
  - Gain intuition for different notions of similarity and practice finding similar documents.
  - Explore the tradeoffs with representing documents using raw word counts and TF-IDF
  - Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obama’s page.


### Implementing Locality Sensitive Hashing from scratch

Locality Sensitive Hashing (LSH) provides for a fast, efficient approximate nearest neighbor search. 
The algorithm scales well with respect to the number of data points as well as dimensions.
  - Implement the LSH algorithm for approximate nearest neighbor search
  - Examine the accuracy for different documents by comparing against brute force search, and also contrast runtimes
  - Explore the role of the algorithm’s tuning parameters in the accuracy of the method


<hr />

Addition;
  - <em>TODO...</em> complete port to [Julia](https://www.julialang.org/)

Outcome:
  - [Jupyter Notebook/PA1](https://github.com/pascal-p/ML_UW_Spec/blob/main/C04/w02/C04w02_nb_pa1.ipynb)
  - [Jupyter Notebook/PA2](https://github.com/pascal-p/ML_UW_Spec/blob/main/C04/w02/C04w02_nb_pa2.ipynb)

<hr />
<p><sub><em>Feb.-Mar. 2021 Corto Inc</sub></em></p>
