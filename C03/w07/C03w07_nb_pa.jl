### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 33bd93ec-77d9-11eb-2da8-87271f8b30c7
begin
  using Pkg
  Pkg.activate("MLJ_env", shared=true)

  # using MLJ
  using CSV
  using DataFrames
  using PlutoUI
  using Test
  using Printf
  using Plots
  using LinearAlgebra
end


# ╔═╡ 6d69b148-77d9-11eb-04f5-034603897c31
begin
  using JSON

  important_words = open("../../ML_UW_Spec/C03/data/important_words.json", "r") do f
      JSON.parse(f)
  end;

  length(important_words)
end

# ╔═╡ 0f6bb770-77df-11eb-0f1f-4dddbd7e2498
begin
	include("./text_proc.jl");
	include("./utils.jl");
end

# ╔═╡ baa8ca1c-77d8-11eb-2e26-3b2445acbfa1
md"""
## Training Logistic Regression via Stochastic Gradient Ascent

The goal of this notebook is to implement a logistic regression classifier using stochastic gradient ascent. We will:
  - Extract features from Amazon product reviews.
  - Convert an SFrame into a NumPy array.
  - Write a function to compute the derivative of log likelihood function with respect to a single coefficient.
  - Implement stochastic gradient ascent.
  - Compare convergence of stochastic gradient ascent with that of batch gradient ascent.

"""

# ╔═╡ 4218be80-7e19-11eb-1706-15ca470f7cf2
begin
	const DF = Union{DataFrame, SubDataFrame}
	const MV = Union{Matrix, Vector} 
end

# ╔═╡ 3ad4770e-77d9-11eb-3a93-addf3e427025
md"""
### Load review dataset

For this assignment, we will use a subset of the Amazon product review dataset. The subset was chosen to contain similar numbers of positive and negative reviews, as the original dataset consisted primarily of positive reviews.
"""

# ╔═╡ 6e264178-77d9-11eb-33c1-19462dde66e2
begin
  products = train = CSV.File("../../ML_UW_Spec/C03/data/amazon_baby_subset.csv";
    header=true) |> DataFrame;
  size(products)
end

# ╔═╡ ba7417fe-7e18-11eb-1706-15ca470f7cf2
describe(products, :eltype, :nmissing, :first => first)

# ╔═╡ 36f2c616-77de-11eb-03b4-3b92a532949a
begin
	## how many missing values for the column :review?
	nmiss₀ = sum(ismissing.(products.review))
	
	## in-place replace(ment) of missing values for review  with empty string
	replace!(products.review, missing =>"");
	
	nmiss₁ = sum(ismissing.(products.review));
	(missing_before_repl=nmiss₀, missing_after_repl=nmiss₁)
end

# ╔═╡ d7b9c812-77de-11eb-0ebb-6bb75de9725b
## no more missing values for column :review
@test sum(ismissing.(products.review)) == 0

# ╔═╡ 6e09ca34-77d9-11eb-1a99-6578f9141004
md"""
One column of this dataset is 'sentiment', corresponding to the class label with +1 indicating a review with positive sentiment and -1 indicating one with negative sentiment.
"""

# ╔═╡ 6ded8978-77d9-11eb-3d94-e7507a85197e
products.sentiment[1:20]

# ╔═╡ 6dd12c56-77d9-11eb-0350-1dfc08d17dd6
md"""
Let us quickly explore more of this dataset. The 'name' column indicates the name of the product. Here we list the first 10 products in the dataset. We then count the number of positive and negative reviews.
"""

# ╔═╡ 6db87918-77d9-11eb-2082-edecdf8b7576
first(products, 10).name

# ╔═╡ 6d316f86-77d9-11eb-107b-1bf5708655d5
md"""
Now, we will perform 2 simple data transformations:

  1. Remove punctuation using Python's built-in string functionality.
  2. Compute word counts (only for important_words)

"""

# ╔═╡ 734ed4d4-77df-11eb-3dab-e5c10b4cda62
md"""
###### Step 1.
"""

# ╔═╡ 0f528d0e-77df-11eb-1415-2fb0e9d6792d
begin
	products[!, :review_clean] = remove_punctuation.(products[:, :review]);
	select!(products, Not(:review));
	
	names(products)
end

# ╔═╡ 0f1c6b16-77df-11eb-2197-93c4a82cc135
md"""
###### Step 2.

For each word in `important_words`, we compute a count for the number of times the word occurs in the review. We will store this count in a separate column (one for each word). The result of this feature processing is a single column for each word in `important_words` which keeps a count of the number of times the respective word occurs in the review text.

"""

# ╔═╡ 0057a50e-77e0-11eb-29d6-85c2dac67743
wc(text::String, word::String) = length(findall(word, text))

# ╔═╡ 51a94e8e-77e3-11eb-04c8-d1389b1f5856
function addcols!(df, words)
  for word ∈ words
    df[!, word] = wc.(df.review_clean, word)
  end
end

# ╔═╡ 0f03e474-77df-11eb-2791-f165fd011737
begin
	addcols!(products, important_words);
	length(names(products))
end

# ╔═╡ d39040fe-78d0-11eb-236d-15b506adaffd
md"""
### Train-Validation split

We split the data into a train-validation split with 90% of the data in the training set and 20% of the data in the validation set.
"""

# ╔═╡ 005dc370-78d1-11eb-0fc0-039f1017af8e
begin
	train_data, validation_data = train_test_split(products; split=0.9, seed=42);
	size(train_data), size(validation_data)
end

# ╔═╡ 682a78ba-78d1-11eb-1706-15ca470f7cf2
md"""
### Using Julia Arrays
"""

# ╔═╡ 0e383a04-77df-11eb-2ec9-c5aeffc4b131
function get_data(df::DF, features, output)
  s_features = [Symbol(f) for f ∈ features]
  df[:, :intercept] .= 1.0
  s_features = [:intercept, s_features...]
  X_matrix = convert(Matrix, select(df, s_features))
  y = df[!, output]

  (X_matrix, Vector{eltype(y)}(y))
end

# ╔═╡ 7ad4c752-78d1-11eb-267b-236c60163a33
begin
	# need a copy to add intercept, train_data is actually a view
	feature_matrix_train, sentiment_train = get_data(copy(train_data),
		important_words, :sentiment);
	# same remark as above concerning validation_data
	feature_matrix_valid, sentiment_valid = get_data(copy(validation_data), 
		important_words, :sentiment);
	
	size(feature_matrix_train), size(sentiment_train)
end

# ╔═╡ b093b9ac-78d1-11eb-3a2e-ffb6e1102b1f
size(feature_matrix_valid), size(sentiment_valid)

# ╔═╡ 54c0f8f2-7e1b-11eb-1706-15ca470f7cf2
md"""
**Quiz Question**: In Module 3 assignment, there were 194 features (an intercept + one feature for each of the 193 important words). <br />
In this assignment, we will use stochastic gradient ascent to train the classifier using logistic regression. How does the changing of the solver to stochastic gradient ascent affect the number of features?
  
  - [ ] Increases
  - [ ] Decreases
  - [x] Stays the same

"""

# ╔═╡ fd23d180-77e6-11eb-2091-af1e84d8fbcf
md"""
#### Building on logistic regression

Recall from lecture that the link function is given by:

$$P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))},$$

where the feature vector $h(\mathbf{x}_i)$ represents the word counts of `important_words` in the review  $\mathbf{x}_i$.

We will use the same code as in Module 3 assignment to make probability predictions, since this part is not affected by using stochastic gradient ascent as a solver. Only the way in which the coefficients are learned is affected by using stochastic gradient ascent as a solver.
"""

# ╔═╡ fd120ad4-77e6-11eb-15ea-bb7d7ea35587
function predict_probability(feature_matrix::MV, coeffs)
	@assert size(coeffs)[2] == 1  # expect a column vector not an array
  	1. ./ (1. .+ exp.(-(feature_matrix * coeffs)))
end

# ╔═╡ fcf1dd2e-77e6-11eb-0619-e7fd2a622877
md"""
#### Derivative of log likelihood with respect to a single coefficient

Let us now work on making minor changes to how the derivative computation is performed for logistic regression.

Recall from the lectures and Module 3 assignment that for logistic regression, **the derivative of log likelihood with respect to a single coefficient** is as follows:

$$\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right)$$

In Module 3 assignment, we wrote a function to compute the derivative of log likelihood with respect to a single coefficient $w_j$. The function accepts the following two parameters:
 * `errors` vector containing $(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w}))$ for all $i$
 * `feature` vector containing $h_j(\mathbf{x}_i)$  for all $i$
 
Complete the following code block:
"""

# ╔═╡ d6976230-78d2-11eb-1fb7-dd1259c9b9c8
feature_derivative(errors, feature) = dot(errors, feature) 

# ╔═╡ 0cea789c-78d4-11eb-36c4-c794c5e1a816
md"""
**Note**. We are not using regularization in this assignment, but, as discussed in the optional video, stochastic gradient can also be used for regularized logistic regression.
"""

# ╔═╡ d6157f9a-7e1b-11eb-267b-236c60163a33
md"""
To verify the correctness of the gradient computation, we provide a function for computing average log likelihood (which we recall from the last assignment was a topic detailed in an advanced optional video, and used here for its numerical stability).

To track the performance of stochastic gradient ascent, we provide a function for computing **average log likelihood**. 

$$\ell\ell_A(\mathbf{w}) = {\color{red}{\frac{1}{N}}} \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big)$$

**Note** that we made one tiny modification to the log likelihood function (called `log_likelihood`) in our earlier assignments. We added a $\color{red}{1/N}$ term which averages the log likelihood accross all data points. The $\color{red}{1/N}$ term makes it easier for us to compare stochastic gradient ascent with batch gradient ascent. We will use this function to generate plots that are similar to those you saw in the lecture.
"""

# ╔═╡ 70a490b2-78d3-11eb-3acd-9bc21f53815e
function avg_log_likelihood(f_matrix::MV, sentiment, coeffs)
	indic = sentiment .== 1
 	scores = f_matrix * coeffs
  	log_exp = log.(1. .+ exp.(-scores))
 	mask = isinf.(log_exp)
  	log_exp[mask] = -scores[mask]
	sum((indic .- 1) .* scores .- log_exp) / length(f_matrix)
end

# ╔═╡ 191a7a84-7e1c-11eb-0389-5978263881b6
md"""
**Quiz Question:** Recall from the lecture and the earlier assignment, the log likelihood (without the averaging term) is given by 

$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big)$$

How are the functions $\ell\ell(\mathbf{w})$ and $\ell\ell_A(\mathbf{w})$ related?
  - answer: $\ell\ell_A(\mathbf{w}) = \frac{1}{N} \times \ell\ell(\mathbf{w})$
"""

# ╔═╡ f6a97742-7e21-11eb-2a06-cb616b0b43d8
md"""
#### Modifying the derivative for stochastic gradient ascent

Recall from the lecture that the gradient for a single data point $\color{red}{\mathbf{x}_i}$ can be computed using the following formula:

```math
\frac{\partial\ell_{\color{red}{i}}(\mathbf{w})}{\partial w_j} = h_j({\color{red}{\mathbf{x}_i}})

\left(
  \mathbf{1}
  [y_{\color{red}{i}} = +1] - P(y_{\color{red}{i}} = +1|{\color{red}{\mathbf{x}_i}},\mathbf{w}) 
  \right)
```

**Computing the gradient for a single data point**

Do we really need to re-write all our code to modify $\partial\ell(\mathbf{w})/\partial w_j$ to $\partial\ell_{\color{red}{i}}(\mathbf{w})/{\partial w_j}$? 


Thankfully **No!**. we can access $\mathbf{x}_i$ in the training data using `feature_matrix_train[i:i, :]`
and $y_i$ in the training data using `sentiment_train[i:i]`. We can compute $\partial\ell_{\color{red}{i}}(\mathbf{w})/\partial w_j$ by re-using **all the code** written in **feature_derivative** and **predict_probability**.


We compute $\partial\ell_{\color{red}{i}}(\mathbf{w})/\partial w_j$ using the following steps:
* First, compute $P(y_i = +1 | \mathbf{x}_i, \mathbf{w})$ using the **predict_probability** function with `feature_matrix_train[i:i, :]` as the first parameter.
* Next, compute $\mathbf{1}[y_i = +1]$ using `sentiment_train[i:i]`.
* Finally, call the **feature_derivative** function with `feature_matrix_train[i:i,  j]` as one of the parameters. 

Let us follow these steps for `j=1` and `i=10`:
"""

# ╔═╡ ba36988a-7e1c-11eb-12f9-1f5e36d0706d
begin
	const N_FEATURES = size(feature_matrix_train)[2]
	ϵ = 1e-8
	j₀ = 1                        # Feature number
	i₀ = 10                       # Data point number
	coeffs₀ = zeros(Float64, (N_FEATURES, 1)) 
	
	preds₀ = predict_probability(feature_matrix_train[i₀:i₀, :], coeffs₀)
	indic₀ = sentiment_train[i₀:i₀] .== 1
	errors₀ = indic₀ - preds₀
	
	∇_single_dp = feature_derivative(errors₀, feature_matrix_train[i₀:i₀, j₀])

	@test abs(∇_single_dp - 0.5) ≤ ϵ
end

# ╔═╡ ba1a42c0-7e1c-11eb-0209-6533fe570748
size(∇_single_dp)  # () means scalar

# ╔═╡ ba03f9ac-7e1c-11eb-1fb7-dd1259c9b9c8
md"""
**Quiz Question:** The code block above computed $\partial\ell_{\color{red}{i}}(\mathbf{w})/{\partial w_j}$ for `j = 1` and `i = 10`.  Is $\partial\ell_{\color{red}{i}}(\mathbf{w})/{\partial w_j}$ a scalar or a 194-dimensional vector?
  - scalar 
"""

# ╔═╡ fd15b4ae-7e23-11eb-10cc-0deed0adb371
md"""
#### Modifying the derivative for using a batch of data points

Stochastic gradient estimates the ascent direction using 1 data point, while gradient uses $N$ data points to decide how to update the the parameters.  In an optional video, we discussed the details of a simple change that allows us to use a **mini-batch** of $B \leq N$ data points to estimate the ascent direction. This simple approach is faster than regular gradient but less noisy than stochastic gradient that uses only 1 data point. Although we encorage you to watch the optional video on the topic to better understand why mini-batches help stochastic gradient, in this assignment, we will simply use this technique, since the approach is very simple and will improve your results.

Given a mini-batch (or a set of data points) $\mathbf{x}_{i}, \mathbf{x}_{i+1} \ldots \mathbf{x}_{i+B}$, the gradient function for this mini-batch of data points is given by:

```math
{\color{red}{\sum_{s = i}^{i+B}}} \frac{\partial\ell_{s}}{\partial w_j} = {\color{red}{\sum_{s = i}^{i + B}}} h_j(\mathbf{x}_s)\left(\mathbf{1}[y_s = +1] - P(y_s = +1 | \mathbf{x}_s, \mathbf{w})\right)
```


**Computing the gradient for a "mini-batch" of data points**

Using NumPy, we access the points $\mathbf{x}_i, \mathbf{x}_{i+1} \ldots \mathbf{x}_{i+B}$ in the training data using `feature_matrix_train[i:i+B,:]`
and $y_i$ in the training data using `sentiment_train[i:i+B]`. 

We can compute ${\color{red}{\sum_{s = i}^{i+B}}} \partial\ell_{s}/\partial w_j$ easily as follows:
"""

# ╔═╡ fcf0e0b4-7e23-11eb-1499-3558c5ea680c
begin
	B₁ = 10  # Mini-batch size
	coeffs₁ = zeros(Float64, (N_FEATURES, 1)) 
	
	preds₁ = predict_probability(feature_matrix_train[i₀:i₀ + B₁, :], coeffs₁)
	indic₁ = sentiment_train[i₀:i₀ + B₁] .== 1
	errors₁ = indic₁ - preds₁
	
	∇_single_mb = feature_derivative(errors₁, feature_matrix_train[i₀:i₀ + B₁, j₀])

	@test abs(∇_single_mb + 1.5) ≤ ϵ
end

# ╔═╡ b98e0666-7e1c-11eb-3a2e-ffb6e1102b1f
size(∇_single_mb)  # () means scalar

# ╔═╡ f2193cb4-7e24-11eb-27e4-55ec5250ae23
size(feature_matrix_train)[1]

# ╔═╡ e1508b4e-7e24-11eb-36ac-8d6c7b6cdaf9
md"""
**Quiz Question:** The code block above computed 
${\color{red}{\sum_{s = i}^{i+B}}}\partial\ell_{s}(\mathbf{w})/{\partial w_j}$ 
for `j = 10`, `i = 10`, and `B = 10`. Is this a scalar or a 194-dimensional vector?
  - scalar

**Quiz Question:** For what value of `B` is the term
${\color{red}{\sum_{s = 1}^{B}}}\partial\ell_{s}(\mathbf{w})/\partial w_j$
the same as the full gradient $\partial\ell(\mathbf{w})/{\partial w_j}$? <br />
Hint: consider the training set we are using now.
  -  B = size(feature_matrix_train)[1] == 47765
"""

# ╔═╡ 1e90e260-7e25-11eb-37ac-a320a56a628e
md"""
#### Averaging the gradient across a batch

It is a common practice to normalize the gradient update rule by the batch size B:

```math
\frac{\partial\ell_{\color{red}{A}}(\mathbf{w})}{\partial w_j} \approx {\color{red}{\frac{1}{B}}} {\sum_{s = i}^{i + B}} h_j(\mathbf{x}_s)\left(\mathbf{1}[y_s = +1] - P(y_s = +1 | \mathbf{x}_s, \mathbf{w})\right)
```

In other words, we update the coefficients using the **average gradient over data points** (instead of using a summation). By using the average gradient, we ensure that the magnitude of the gradient is approximately the same for all batch sizes. This way, we can more easily compare various batch sizes of stochastic gradient ascent (including a batch size of **all the data points**), and study the effect of batch size on the algorithm as well as the choice of step size.


## Implementing stochastic gradient ascent

Now we are ready to implement our own logistic regression with stochastic gradient ascent. Complete the following function to fit a logistic regression model using gradient ascent:
"""

# ╔═╡ 1e437548-7e25-11eb-3e76-49a4a2f4094f
function shuffle_data(f_matrix::MV, sentiment::Vector; seed=42)  
    "Set specific seed to produce consistent results"
    n = size(f_matrix)[1]
	perm = randperm(MersenneTwister(seed), n)
    f_matrix = f_matrix[perm, :]
    sentiment = sentiment[perm]
    (f_matrix, sentiment)
end

# ╔═╡ 970fba24-77ed-11eb-1a3e-e769d6049078
function logistic_regression_SG(f_matrix::MV, sentiment::Vector, init_coeffs; 
		η=1e-7, batch_sz=100, max_iter=200)
	#
	coeffs = copy(init_coeffs)
	n = size(f_matrix)[1]
	f_matrix, sentiment = shuffle_data(f_matrix, sentiment)
	ll_all = Float64[]
	ix = 1
  	for it ∈ 1:max_iter
		n_ix = ix + batch_sz - 1
		
    	preds = predict_probability(f_matrix[ix:n_ix, :], coeffs)
    	indic = sentiment[ix:n_ix] .== 1
    	errors = indic .- preds

    	for jx ∈ 1:length(coeffs)
	  		is_intercept = (jx == 1)
      		∂c = feature_derivative(errors, f_matrix[ix:n_ix, jx])
      		coeffs[jx] += (1. / batch_sz) * η * ∂c
    	end

		ll = avg_log_likelihood(f_matrix[ix:n_ix, :], sentiment[ix:n_ix], coeffs)
        push!(ll_all, ll)
		
    	if it ≤ 15 || (it ≤ 100 && it % 10 == 0) || (it ≤ 1000 && it % 100 == 0) ||
      	(it ≤ 10000 && it % 1000 == 0) || it % 10000 == 0
      		@printf("iter: %2d: avg log likelihood (of data points in batch [%4d:%4d]): %.8f\n", it, ix, n_ix, ll)
    	end
		
		ix += batch_sz - 1
		
		if ix + batch_sz - 1 > n
			f_matrix, sentiment = shuffle_data(f_matrix, sentiment)
			ix = 1
		end
  	end

  	(coeffs, ll_all)
end

# ╔═╡ e6ef9818-7e26-11eb-08c1-13392c483a14
md"""
*Remark*: In practice, the final set of coefficients is rarely used; it is better to use the average of the last K sets of coefficients instead, where K should be adjusted depending on how fast the log likelihood oscillates around the optimum.
"""

# ╔═╡ e6d10c2c-7e26-11eb-05fa-d953b60010ff
md"""
##### Checkpoint

The following cell tests your stochastic gradient ascent function using a toy dataset consisting of two data points. If the test does not pass, make sure you are normalizing the gradient update rule correctly.
"""

# ╔═╡ e6b80b32-7e26-11eb-2b8f-a9b22b5c46fe
begin
	f_matrix₂ = Float64[1. 2. -1.; 1. 0. 1.]
	sentiment₂ = Int[1, -1]
	init_coeffs₂=zeros(Float64, (3, 1))
	
	coeffs₂, log_ll₂ = logistic_regression_SG(f_matrix₂, sentiment₂, init_coeffs₂; 
		η=1., batch_sz=2, max_iter=2)

	ϵ₁ = 1e-3
	exp_coeffs₂ = [-0.0975576 0.682426 -0.779983]
	exp_log_ll₂ = [-0.112582 -0.0781844]

	# coeffs₂, log_ll₂
	@test all(((cₐ, cₑ)=t) -> abs(cₐ - cₑ) ≤ ϵ₁, zip(coeffs₂, exp_coeffs₂)) 
	@test all(((llₐ, llₑ)=t) -> abs(llₐ - llₑ) ≤ ϵ₁, zip(log_ll₂, exp_log_ll₂))
end

# ╔═╡ e696bce6-7e26-11eb-218d-ff1d2947d397
with_terminal() do
	println("----------------------------------------------------------------")
	println("Coefficients learned: $(coeffs₂)")
	println("Average log likelihood per-iteration: $(log_ll₂)")
	println("----------------------------------------------------------------")
	print("Test passed!")
end

# ╔═╡ 3cbf62b8-7e2d-11eb-3a2e-ffb6e1102b1f
md"""
#### Compare convergence behavior of stochastic gradient ascent

For the remainder of the assignment, we will compare stochastic gradient ascent against batch gradient ascent. For this, we need a reference implementation of batch gradient ascent. But do we need to implement this from scratch?

**Quiz Question:** For what value of batch size `B` above is the stochastic gradient ascent function `logistic_regression_SG` act as a standard gradient ascent algorithm? 

Hint: consider the training set we are using now.
"""

# ╔═╡ 54f2b0a6-7e2d-11eb-1fb7-dd1259c9b9c8
# answer B should be:
size(train_data)[1]

# ╔═╡ 7f298886-7e2d-11eb-0209-6533fe570748
md"""
#### Running gradient ascent using the stochastic gradient ascent implementation

Instead of implementing batch gradient ascent separately, we save time by re-using the stochastic gradient ascent function we just wrote &mdash; **to perform gradient ascent**, it suffices to set **`batch_size`** to the number of data points in the training data. Yes, we did answer above the quiz question for you, but that is an important point to remember in the future :)

**Small Caveat**. The batch gradient ascent implementation here is slightly different than the one in the earlier assignments, as we now normalize the gradient update rule.

We now **run stochastic gradient ascent** over the `feature_matrix_train` for 10 iterations using:
  - `init_coeffs = zeros(Float64, (194, 1))`
  - `η = 5e-1`
  - `batch_sz = 1`
  - `max_iter = 10`

"""

# ╔═╡ b762f3d6-7e2d-11eb-3906-5ba3566b9ae9
begin
	init_coeffs₃ = zeros(Float64, (N_FEATURES, 1))
	local coeffs₃, log_likelihood₃ 
	
	with_terminal() do
		coeffs₃, log_likelihood₃ = logistic_regression_SG(feature_matrix_train,
			sentiment_train, init_coeffs₃,
			η=5e-1, batch_sz=1, max_iter=10);
	end
end

# ╔═╡ b715ba4e-7e2d-11eb-36c4-c794c5e1a816
md"""
**Quiz Question**. When you set `batch_size = 1`, as each iteration passes, how does the average log likelihood in the batch change?
  - [x] Increases
  - [ ] Decreases
  - [ ] Fluctuates 
"""

# ╔═╡ b6f8ad50-7e2d-11eb-3acd-9bc21f53815e
md"""
Now run **batch gradient ascent** over the `feature_matrix_train` for 200 iterations using:

  - `init_coeffs = zeros(Float64, (194, 1))`
  - `η = 5e-1`
  - `batch_sz = size(feature_matrix_train)[1]`
  - `max_iter = 200`

"""

# ╔═╡ b6e038ec-7e2d-11eb-13d1-8dafac6a1d6a
begin
	# takes a while ≈ 98.6s
	init_coeffs₄ = zeros(Float64, (N_FEATURES, 1))
	coeffs₄, log_likelihood₄ = nothing, nothing
	
	with_terminal() do
		global coeffs₄, log_likelihood₄ = logistic_regression_SG(feature_matrix_train,
			sentiment_train, init_coeffs₄,
			η=5e-1, batch_sz=size(feature_matrix_train)[1], max_iter=200);
	end
end

# ╔═╡ b6c44948-7e2d-11eb-12f9-1f5e36d0706d
md"""
**Quiz Question**. When you set `batch_size = len(feature_matrix_train)`, as each iteration passes, how does the average log likelihood in the batch change?
  - [x] Increases 
  - [ ] Decreases
  - [ ] Fluctuates 
"""

# ╔═╡ 14f788e6-7e30-11eb-27e4-55ec5250ae23
md"""
#### Make "passes" over the dataset

To make a fair comparison betweeen stochastic gradient ascent and batch gradient ascent, we measure the average log likelihood as a function of the number of passes (defined as follows):

$$[\text{\# of passes}] = \frac{[\text{\# of data points touched so far}]}{[\text{size of dataset}]}$$

"""

# ╔═╡ 14dcb3f6-7e30-11eb-36ac-8d6c7b6cdaf9
md"""
**Quiz Question** Suppose that we run stochastic gradient ascent with a batch size of 100. How many gradient updates are performed at the end of two passes over a dataset consisting of 50000 data points?
"""

# ╔═╡ 14c1d310-7e30-11eb-10cc-0deed0adb371
# Answer:
2 * 50_000 ÷ 100

# ╔═╡ 14a70a76-7e30-11eb-1499-3558c5ea680c
md"""
#### Log likelihood plots for stochastic gradient ascent

With the terminology in mind, let us run stochastic gradient ascent for 10 passes. We will use
  - `η=1e-1`
  - `batch_sz=100`
  - `init_coeffs` to all zeros.
"""

# ╔═╡ 148f157e-7e30-11eb-1d8a-1f920b745781
begin
	n_passes₅ = 10
	batch_sz₅ = 100
	n_iter₅ = n_passes₅ * size(feature_matrix_train)[1] ÷ batch_sz₅
	init_coeffs₅ = zeros(Float64, (N_FEATURES, 1))
	coeffs_sgd₅, log_ll_sgd₅ = nothing, nothing
	
	with_terminal() do
		global coeffs_sgd₅, log_ll_sgd₅ = logistic_regression_SG(feature_matrix_train, 	sentiment_train, init_coeffs₅; η=1e-1, batch_sz=batch_sz₅, max_iter=n_iter₅)
	end
end

# ╔═╡ 09444fda-7e31-11eb-3e76-49a4a2f4094f
# TODO plots
log_ll_sgd₅

# ╔═╡ eae04a48-7e31-11eb-08c1-13392c483a14
md"""
#### Smoothing the stochastic gradient ascent curve

The plotted line oscillates so much that it is hard to see whether the log likelihood is improving. In our plot, we apply a simple smoothing operation using the parameter smoothing_window.
The smoothing is simply a moving average of log likelihood over the last smoothing_window "iterations" of stochastic gradient ascent.

"""

# ╔═╡ a4b3fe7a-7e45-11eb-0209-6533fe570748
function convolve(v₁::AbstractVector, v₂::AbstractVector; mode=:valid)
	if mode != :valid
		# other modes: :same, :full - not required here
		throw(ArgumentError("mode $(mode) not implemented"))
	end
	nv₁, nv₂ = length(v₁), length(v₂)
	@assert nv₁ ≤ nv₂
		
	vs₁ = sort(v₁, rev=true)
	nv = zeros(eltype(v₁), nv₂ - nv₁ + 1)
				
	for ix ∈ 1:nv₂ - nv₁ + 1
		nv[ix] = dot(vs₁, v₂[ix:ix+nv₁ - 1]) 
	end
	nv
end

# ╔═╡ c83d6ece-7e47-11eb-12f9-1f5e36d0706d
@test convolve([1.; 2.; 3.], [0.; 1.; 0.5]) == Float64[2.5]

# ╔═╡ 4f1d3fb4-7e48-11eb-13d1-8dafac6a1d6a
@test convolve([3.; 7.], [1.; 2.; 5.; 7.]) == [13.0, 29.0, 56.0]

# ╔═╡ 498064f0-7e4d-11eb-3a2e-ffb6e1102b1f
function apply_conv(ll_all, swin)
	n = length(ll_all)
	ll_all_ma = convolve(ones(eltype(ll_all), swin) ./ swin, ll_all, mode=:valid)
	(ll_all_ma, n)
end

# ╔═╡ 32e899e0-7e45-11eb-1fb7-dd1259c9b9c8
function make_plot(ll_all, len, batch_sz; swin=1, label="", leg=:topright, over=false)
	ll_all_ma, n = apply_conv(ll_all, swin)
	plot(collect(1:n - swin + 1) * batch_sz / len, ll_all_ma, 
		linewidth=3.0, label=label, leg=leg, 
		ylabel="Average log likelihood per data point",
		xlabel="# of passes over data")
end

# ╔═╡ 334dece8-7e4d-11eb-0389-5978263881b6
function make_plot!(ll_all, len, batch_sz; swin=1, label="", leg=:topright)
	ll_all_ma, n = apply_conv(ll_all, swin)
	plot!(collect(1:n - swin + 1) * batch_sz / len, ll_all_ma, 
		linewidth=3.0, label=label, leg=leg, 
		ylabel="Average log likelihood per data point",
		xlabel="# of passes over data")
end

# ╔═╡ d98e2fe0-7e49-11eb-3906-5ba3566b9ae9
make_plot(log_ll_sgd₅, size(feature_matrix_train)[1], batch_sz₅; 
	label="stochastic gradient, step_size=1e-1", leg=:topleft)

# ╔═╡ 11f38c2a-7e4c-11eb-1706-15ca470f7cf2
md"""
###### Smoothing the stochastic gradient ascent curve

The plotted line oscillates so much that it is hard to see whether the log likelihood is improving. In our plot, we apply a simple smoothing operation using the parameter smoothing_window.

The smoothing is simply a moving average of log likelihood over the last smoothing_window (`swin`) "iterations" of stochastic gradient ascent.

"""

# ╔═╡ 1a250d88-7e4c-11eb-267b-236c60163a33
make_plot(log_ll_sgd₅, size(feature_matrix_train)[1], batch_sz₅; 
          swin=30, label="stochastic gradient, step_size=1e-1", leg=:bottomright)

# ╔═╡ eab39b24-7e31-11eb-2b8f-a9b22b5c46fe
md"""
#### Stochastic gradient ascent vs batch gradient ascent

To compare convergence rates for stochastic gradient ascent with batch gradient ascent, we call `make_plot()` multiple times in the same cell.

We are comparing:
  * **stochastic gradient ascent**: `η = 0.1`, `batch_sz=100`
  * **batch gradient ascent**: `step_size = 0.5`, `batch_size=len(feature_matrix_train)`

Write code to run stochastic gradient ascent for 200 passes using:
  - `η=1e-1`
  - `batch_sz=100`
  - `init_coeffs` to all zeros.
"""

# ╔═╡ 042eab34-7e32-11eb-1f51-2729cf921bc5
begin
	batch_sz₆ = 100
	n_passes₆ = 200
	n_iter₆ = n_passes₆ * size(feature_matrix_train)[1] ÷ batch_sz₆
	coeffs_sgd₆, log_ll_sgd₆ = nothing, nothing
	
	with_terminal() do
		global coeffs_sgd₆, log_ll_sgd₆ = logistic_regression_SG(feature_matrix_train, sentiment_train, init_coeffs₅; η=1e-1, batch_sz=batch_sz₆, max_iter=n_iter₆)
	end
end

# ╔═╡ ea95b6a4-7e31-11eb-218d-ff1d2947d397
begin
	make_plot(log_ll_sgd₆, size(feature_matrix_train)[1], batch_sz₆; 
          swin=30, label="batch, step_size=1e-1", leg=:bottomright)
	
	make_plot!(log_likelihood₄, size(feature_matrix_train)[1], 
		size(feature_matrix_train)[1]; 
		swin=30, label="stochastic gradient, step_size=5e-1", leg=:bottomright)
end

# ╔═╡ f360444a-7e41-11eb-1706-15ca470f7cf2
md"""
#### Explore the effects of step sizes on stochastic gradient ascent

In previous sections, we chose step sizes for you. In practice, it helps to know how to choose good step sizes yourself.

To start, we explore a wide range of step sizes that are equally spaced in the log space. Run stochastic gradient ascent with `step_size` set to 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, and 1e2. Use the following set of parameters:
  - `init_coeff all zeros`
  - `batch_sz=100`
  - `max_iter` initialized so as to run 10 passes over the data.
"""

# ╔═╡ 22472020-7e42-11eb-267b-236c60163a33
begin
	n_passes₇ = 10
	batch_sz₇ = 100
	n_iter₇ = n_passes₇ * size(feature_matrix_train)[1] ÷ batch_sz₇
	init_coeffs₇ = zeros(Float64, (N_FEATURES, 1))
	c_sgd, log_ll_sgd = Dict{Float64, Any}(), Dict{Float64, Any}()
	
	const ETA = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
	with_terminal() do
		for η ∈ ETA
			println("\tη: $(η)")
			global c_sgd[η], log_ll_sgd[η] = logistic_regression_SG( 								feature_matrix_train, 
				sentiment_train, init_coeffs₇; 
				η=η, batch_sz=batch_sz₇, max_iter=n_iter₇
			)
		end
	end
end

# ╔═╡ b9985a96-7e42-11eb-3a2e-ffb6e1102b1f
md"""
#### Plotting the log likelihood as a function of passes for each step size

Now, we will plot the change in log likelihood using the `make_plot` for each of the following values of η (cf. cell above):

For consistency, we again apply smoothing_window: `swin=30`.
"""

# ╔═╡ 3940bc4a-7e50-11eb-12f9-1f5e36d0706d
begin
	len = size(feature_matrix_train)[1]
	η = 1e-4
	make_plot(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)

	η = 1e-3
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)

	η = 1e-2
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)

	
	η = 1e-1
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)

	η = 1e0
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)
	
	η = 1e1
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)
	
	η = 1e2
	make_plot!(log_ll_sgd[η], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η)", leg=:bottomright)
end

# ╔═╡ 848d7b2c-7e53-11eb-2a06-cb616b0b43d8
md"""
Now, let us remove the step size η=1e2 and η=1e1 plot the rest of the curves.
"""

# ╔═╡ 8ce3d06e-7e53-11eb-2873-9fd54b4485ea
begin
	η₁ = 1e-4
	make_plot(log_ll_sgd[η₁], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)

	η₁ = 1e-3
	make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)

	η₁ = 1e-2
	make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)
	
	η₁ = 1e-1
	make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)

	η₁ = 1e0
	make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)
	
	# η₁ = 1e1
	# make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
	# 		swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)
	
	# η₁ = 1e2
	# make_plot!(log_ll_sgd[η₁], len, batch_sz₇; 
	# 		swin=30, label="batch, step_size=$(η₁)", leg=:bottomright)
end

# ╔═╡ ebacd9c4-7e53-11eb-1499-3558c5ea680c
begin
	η₂ = 1e1
	make_plot(log_ll_sgd[η₂], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₂)", leg=:bottomright)
	
	η₂ = 1e2
	make_plot!(log_ll_sgd[η₂], len, batch_sz₇; 
			swin=30, label="batch, step_size=$(η₂)", leg=:bottomright)
end

# ╔═╡ dea572a2-7e53-11eb-1d8a-1f920b745781
md"""
**Quiz Question**: Which of the following is the worst step size? Pick the step size that results in the lowest log likelihood in the end.
  1. [ ] 1e-2
  2. [ ] 1e-1
  3. [ ] 1e0
  4. [ ] 1e1
  5. [x] 1e2 (the one we eliminayed...)


**Quiz Question**: Which of the following is the best step size? Pick the step size that results in the highest log likelihood in the end.
  1. [ ] 1e-4
  2. [ ] 1e-2
  3. [x] 1e0
  4. [ ] 1e1
  5. [ ] 1e2
"""

# ╔═╡ Cell order:
# ╟─baa8ca1c-77d8-11eb-2e26-3b2445acbfa1
# ╠═33bd93ec-77d9-11eb-2da8-87271f8b30c7
# ╠═0f6bb770-77df-11eb-0f1f-4dddbd7e2498
# ╠═4218be80-7e19-11eb-1706-15ca470f7cf2
# ╟─3ad4770e-77d9-11eb-3a93-addf3e427025
# ╠═6e264178-77d9-11eb-33c1-19462dde66e2
# ╠═ba7417fe-7e18-11eb-1706-15ca470f7cf2
# ╠═36f2c616-77de-11eb-03b4-3b92a532949a
# ╠═d7b9c812-77de-11eb-0ebb-6bb75de9725b
# ╟─6e09ca34-77d9-11eb-1a99-6578f9141004
# ╠═6ded8978-77d9-11eb-3d94-e7507a85197e
# ╟─6dd12c56-77d9-11eb-0350-1dfc08d17dd6
# ╠═6db87918-77d9-11eb-2082-edecdf8b7576
# ╠═6d69b148-77d9-11eb-04f5-034603897c31
# ╟─6d316f86-77d9-11eb-107b-1bf5708655d5
# ╟─734ed4d4-77df-11eb-3dab-e5c10b4cda62
# ╠═0f528d0e-77df-11eb-1415-2fb0e9d6792d
# ╟─0f1c6b16-77df-11eb-2197-93c4a82cc135
# ╠═0057a50e-77e0-11eb-29d6-85c2dac67743
# ╠═51a94e8e-77e3-11eb-04c8-d1389b1f5856
# ╠═0f03e474-77df-11eb-2791-f165fd011737
# ╟─d39040fe-78d0-11eb-236d-15b506adaffd
# ╠═005dc370-78d1-11eb-0fc0-039f1017af8e
# ╟─682a78ba-78d1-11eb-1706-15ca470f7cf2
# ╠═0e383a04-77df-11eb-2ec9-c5aeffc4b131
# ╠═7ad4c752-78d1-11eb-267b-236c60163a33
# ╠═b093b9ac-78d1-11eb-3a2e-ffb6e1102b1f
# ╟─54c0f8f2-7e1b-11eb-1706-15ca470f7cf2
# ╟─fd23d180-77e6-11eb-2091-af1e84d8fbcf
# ╠═fd120ad4-77e6-11eb-15ea-bb7d7ea35587
# ╟─fcf1dd2e-77e6-11eb-0619-e7fd2a622877
# ╠═d6976230-78d2-11eb-1fb7-dd1259c9b9c8
# ╟─0cea789c-78d4-11eb-36c4-c794c5e1a816
# ╟─d6157f9a-7e1b-11eb-267b-236c60163a33
# ╠═70a490b2-78d3-11eb-3acd-9bc21f53815e
# ╟─191a7a84-7e1c-11eb-0389-5978263881b6
# ╟─f6a97742-7e21-11eb-2a06-cb616b0b43d8
# ╠═ba36988a-7e1c-11eb-12f9-1f5e36d0706d
# ╠═ba1a42c0-7e1c-11eb-0209-6533fe570748
# ╟─ba03f9ac-7e1c-11eb-1fb7-dd1259c9b9c8
# ╟─fd15b4ae-7e23-11eb-10cc-0deed0adb371
# ╠═fcf0e0b4-7e23-11eb-1499-3558c5ea680c
# ╠═b98e0666-7e1c-11eb-3a2e-ffb6e1102b1f
# ╠═f2193cb4-7e24-11eb-27e4-55ec5250ae23
# ╟─e1508b4e-7e24-11eb-36ac-8d6c7b6cdaf9
# ╟─1e90e260-7e25-11eb-37ac-a320a56a628e
# ╠═1e437548-7e25-11eb-3e76-49a4a2f4094f
# ╠═970fba24-77ed-11eb-1a3e-e769d6049078
# ╟─e6ef9818-7e26-11eb-08c1-13392c483a14
# ╟─e6d10c2c-7e26-11eb-05fa-d953b60010ff
# ╠═e6b80b32-7e26-11eb-2b8f-a9b22b5c46fe
# ╠═e696bce6-7e26-11eb-218d-ff1d2947d397
# ╟─3cbf62b8-7e2d-11eb-3a2e-ffb6e1102b1f
# ╠═54f2b0a6-7e2d-11eb-1fb7-dd1259c9b9c8
# ╟─7f298886-7e2d-11eb-0209-6533fe570748
# ╠═b762f3d6-7e2d-11eb-3906-5ba3566b9ae9
# ╟─b715ba4e-7e2d-11eb-36c4-c794c5e1a816
# ╟─b6f8ad50-7e2d-11eb-3acd-9bc21f53815e
# ╠═b6e038ec-7e2d-11eb-13d1-8dafac6a1d6a
# ╟─b6c44948-7e2d-11eb-12f9-1f5e36d0706d
# ╟─14f788e6-7e30-11eb-27e4-55ec5250ae23
# ╟─14dcb3f6-7e30-11eb-36ac-8d6c7b6cdaf9
# ╠═14c1d310-7e30-11eb-10cc-0deed0adb371
# ╟─14a70a76-7e30-11eb-1499-3558c5ea680c
# ╠═148f157e-7e30-11eb-1d8a-1f920b745781
# ╠═09444fda-7e31-11eb-3e76-49a4a2f4094f
# ╟─eae04a48-7e31-11eb-08c1-13392c483a14
# ╠═a4b3fe7a-7e45-11eb-0209-6533fe570748
# ╠═c83d6ece-7e47-11eb-12f9-1f5e36d0706d
# ╠═4f1d3fb4-7e48-11eb-13d1-8dafac6a1d6a
# ╠═498064f0-7e4d-11eb-3a2e-ffb6e1102b1f
# ╠═32e899e0-7e45-11eb-1fb7-dd1259c9b9c8
# ╠═334dece8-7e4d-11eb-0389-5978263881b6
# ╠═d98e2fe0-7e49-11eb-3906-5ba3566b9ae9
# ╟─11f38c2a-7e4c-11eb-1706-15ca470f7cf2
# ╠═1a250d88-7e4c-11eb-267b-236c60163a33
# ╟─eab39b24-7e31-11eb-2b8f-a9b22b5c46fe
# ╠═042eab34-7e32-11eb-1f51-2729cf921bc5
# ╠═ea95b6a4-7e31-11eb-218d-ff1d2947d397
# ╟─f360444a-7e41-11eb-1706-15ca470f7cf2
# ╠═22472020-7e42-11eb-267b-236c60163a33
# ╟─b9985a96-7e42-11eb-3a2e-ffb6e1102b1f
# ╠═3940bc4a-7e50-11eb-12f9-1f5e36d0706d
# ╟─848d7b2c-7e53-11eb-2a06-cb616b0b43d8
# ╠═8ce3d06e-7e53-11eb-2873-9fd54b4485ea
# ╠═ebacd9c4-7e53-11eb-1499-3558c5ea680c
# ╟─dea572a2-7e53-11eb-1d8a-1f920b745781
