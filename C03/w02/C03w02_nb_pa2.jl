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
  # using Plots
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
include("./text_proc.jl");

# ╔═╡ 1c52f3b6-78d1-11eb-3077-bf6c471251c8
include("./utils.jl")

# ╔═╡ baa8ca1c-77d8-11eb-2e26-3b2445acbfa1
md"""
## Logistic Regression with L2 regularization

The goal of this second notebook is to implement your own logistic regression classifier with L2 regularization. We will do the following:

 - Extract features from Amazon product reviews.
 - Turn a DataFrame into a Julia array.
 - Write a function to compute the derivative of log likelihood function with an L2 penalty with respect to a single coefficient.
 - Implement gradient ascent with an L2 penalty.
 - Empirically explore how the L2 penalty can ameliorate overfitting.

"""

# ╔═╡ 3ad4770e-77d9-11eb-3a93-addf3e427025
md"""
### Load review dataset

For this assignment, we will use a subset of the Amazon product review dataset. The subset was chosen to contain similar numbers of positive and negative reviews, as the original dataset consisted primarily of positive reviews.
"""

# ╔═╡ 6e264178-77d9-11eb-33c1-19462dde66e2
begin
  products = train = CSV.File("../../ML_UW_Spec/C03/data/amazon_baby_subset.csv";
    header=true) |> DataFrame;

  first(products, 3)
end

# ╔═╡ 75f2e820-77de-11eb-19eb-a5a5977b5257
size(products)

# ╔═╡ 36f2c616-77de-11eb-03b4-3b92a532949a
## how many missing values for the column :review?
sum(ismissing.(products.review))

# ╔═╡ 6d4ef592-77d9-11eb-3844-1bea7dc71094
## in-place replace(ment) of missing values for review  with empty string
replace!(products.review, missing =>"");

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
end

# ╔═╡ 0f362fec-77df-11eb-02a6-210a58c2839b
names(products)

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
@time addcols!(products, important_words)

# ╔═╡ 0ee68550-77df-11eb-0a6e-c5bf5e3fd9ac
length(names(products))

# ╔═╡ 0ecc0054-77df-11eb-21a4-3f677548ecf9
products.perfect, length(products.perfect)

# ╔═╡ 0eb50d9a-77df-11eb-0b7d-557ab3ea2c81
md"""
Now, write some code to compute the number of product reviews that contain the word perfect.
"""

# ╔═╡ d39040fe-78d0-11eb-236d-15b506adaffd
md"""
### Train-Validation split

We split the data into a train-validation split with 80% of the data in the training set and 20% of the data in the validation set.

**Note:** In previous assignments, we have called this a **train-test split**. However, the portion of data that we don't train on will be used to help **select model parameters**. Thus, this portion of data should be called a **validation set**. Recall that examining performance of various potential models (i.e. models with different parameters) should be on a validation set, while evaluation of selected model should always be on a test set.
"""

# ╔═╡ 005dc370-78d1-11eb-0fc0-039f1017af8e
train_data, validation_data = train_test_split(products; split=0.8, seed=42);

# ╔═╡ 004d22cc-78d1-11eb-10ce-09eb125a96b3
size(train_data), size(validation_data) 

# ╔═╡ 682a78ba-78d1-11eb-1706-15ca470f7cf2
md"""
### Using Julia Arrays
"""

# ╔═╡ 0e383a04-77df-11eb-2ec9-c5aeffc4b131
function get_data(df::DF, features, output) where {DF <: Union{DataFrame, SubDataFrame}}
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
end

# ╔═╡ 9f29e2b8-78d1-11eb-0389-5978263881b6
size(feature_matrix_train), size(sentiment_train)

# ╔═╡ b093b9ac-78d1-11eb-3a2e-ffb6e1102b1f
size(feature_matrix_valid), size(sentiment_valid)

# ╔═╡ fd23d180-77e6-11eb-2091-af1e84d8fbcf
md"""
#### Building on logistic regression with no L2 penalty assignment

Recall from lecture that the link function is given by:

$$P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))},$$

where the feature vector $h(\mathbf{x}_i)$ represents the word counts of `important_words` in the review  $\mathbf{x}_i$.

Implement the `predict_probability` function that implements the link function:
"""

# ╔═╡ fd120ad4-77e6-11eb-15ea-bb7d7ea35587
function predict_probability(feature_matrix::Matrix, coeffs)
  1. ./ (1. .+ exp.(-(feature_matrix * coeffs)))
end

# ╔═╡ fcf1dd2e-77e6-11eb-0619-e7fd2a622877
md"""
### Adding  L2 penalty

Let us now work on extending logistic regression with L2 regularization. As discussed in the lectures, the L2 regularization is particularly useful in preventing overfitting. In this assignment, we will explore L2 regularization in detail.

Recall from lecture and the previous assignment that for logistic regression without an L2 penalty, the derivative of the log likelihood function is:

$$\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right)$$

**Adding L2 penalty to the derivative** 

It takes only a small modification to add a L2 penalty. All terms indicated in **red** refer to terms that were added due to an **L2 penalty**.

  - Recall from the lecture that the link function is still the sigmoid:
$$P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))},$$

  - We add the L2 penalty term to the per-coefficient derivative of log likelihood:
$$\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right) \color{red}{-2\lambda w_j }$$

The **per-coefficient derivative for logistic regression with an L2 penalty** is as follows:

$$\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right) \color{red}{-2\lambda w_j }$$

and for the intercept term, we have
$$\frac{\partial\ell}{\partial w_0} = \sum_{i=1}^N h_0(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right)$$

"""

# ╔═╡ d6f4542a-78d2-11eb-0209-6533fe570748
md"""
Note: As we did in the Regression course, we do not apply the L2 penalty on the intercept. A large intercept does not necessarily indicate overfitting because the intercept is not associated with any particular feature.

Write a function that computes the derivative of log likelihood with respect to a single coefficient $w_j$. Unlike its counterpart in the last assignment, the function accepts five arguments:

  - `errors` vector containing $(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w}))$ for all $i$
  - `feature` vector containing $h_j(\mathbf{x}_i)$  for all $i$
  -  `coefficient` containing the current value of coefficient $w_j$.
  -  `l2_penalty` representing the L2 penalty constant $\lambda$
  -  `feature_is_constant` telling whether the $j$-th feature is constant or not.
"""

# ╔═╡ d6976230-78d2-11eb-1fb7-dd1259c9b9c8
function feature_derivative_with_l2(errors, feature, coeff, l2_penalty; 
		is_intercept=false) 
    deriv = dot(errors, feature) 
    ## add L2 penalty term for any feature that isn't the intercept.
    !is_intercept && (deriv -= 2 * l2_penalty * coeff)
	deriv
end

# ╔═╡ 4870ab12-78d3-11eb-12f9-1f5e36d0706d
md"""
**Quiz Question:** In the code above, was the intercept term regularized?
 - No (because of flag `feature_is_constant`)
"""

# ╔═╡ 665516c2-78d3-11eb-13d1-8dafac6a1d6a
md"""
To verify the correctness of the gradient ascent algorithm, we provide a function for computing log likelihood (which we recall from the last assignment was a topic detailed in an advanced optional video, and used here for its numerical stability).


$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big) \color{red}{-\lambda\|\mathbf{w}\|_2^2}$$
"""

# ╔═╡ 70a490b2-78d3-11eb-3acd-9bc21f53815e
function log_likelihood_with_l2(f_matrix, sentiment, coeffs, l2_penalty)
	indic = sentiment .== 1
 	scores = f_matrix * coeffs
  	log_exp = log.(1. .+ exp.(-scores))
 	# mask = isinf.(log_exp)
  	# log_exp[mask] = -scores[mask]
	sum((indic .- 1) .* scores .- log_exp) - l2_penalty * sum(coeffs[2:end] .^ 2)
end

# ╔═╡ 0cea789c-78d4-11eb-36c4-c794c5e1a816
md"""
**Quiz Question:** Does the term with L2 regularization increase or decrease $\ell\ell(\mathbf{w})$?
  - should decrease
"""

# ╔═╡ 1a7b28f8-78d4-11eb-3906-5ba3566b9ae9
md"""
The `logistic_regression` function looks almost like the one in the last assignment, with a minor modification to account for the L2 penalty. Fill in the code below to complete this modification.
"""

# ╔═╡ 970fba24-77ed-11eb-1a3e-e769d6049078
function logistic_regression_with_l2(f_matrix::Matrix, sentiment::Vector, 
		init_coeffs, l2_penalty; 
		η=1e-7, max_iter=200)
  coeffs = copy(init_coeffs)

  for ix ∈ 1:max_iter
    preds = predict_probability(f_matrix, coeffs)
    indic = sentiment .== 1
    errors = indic .- preds

    for jx ∈ 1:length(coeffs)
	  is_intercept = (jx == 1)
      ∂c = feature_derivative_with_l2(errors, f_matrix[:, jx], coeffs[jx], 
				l2_penalty; is_intercept)
      coeffs[jx] += η * ∂c
    end

    if ix ≤ 15 || (ix ≤ 100 && ix % 10 == 0) || (ix ≤ 1000 && ix % 100 == 0) ||
      (ix ≤ 10000 && ix % 1000 == 0) || ix % 10000 == 0
      lp = log_likelihood_with_l2(f_matrix, sentiment, coeffs, l2_penalty)
      @printf("iter: %d / %d: log likelihood of observed labels: %.8f\n",
        ceil(Int, log10(max_iter)), ix, lp)
    end
  end

  coeffs
end

# ╔═╡ 972e0c72-77ed-11eb-0ae7-c127694faf89
md"""
### Explore effects of L2 regularization

Now that we have written up all the pieces needed for regularized logistic regression, let's explore the benefits of using **L2 regularization** in analyzing sentiment for product reviews. **As iterations pass, the log likelihood should increase**.

Below, we train models with increasing amounts of regularization, starting with no L2 penalty, which is equivalent to our previous logistic regression implementation.
"""

# ╔═╡ afa2bff4-78d4-11eb-10cc-0deed0adb371
function run_model(;matrix_tr=feature_matrix_train, 
		s_train=sentiment_train, 
		init_coeffs=zeros(Float64, (194, 1)), 
		l2_penalty=0, η=5e-6, max_iter=501)
	
	logistic_regression_with_l2(matrix_tr, s_train, init_coeffs, l2_penalty; η, 
		max_iter)
end

# ╔═╡ af858fea-78d4-11eb-1499-3558c5ea680c
begin
	hcoeffs = Dict{String, Matrix}()

	const KV_LIST =  [(0, "coefficients [L2=0]"), 
		(4, "coefficients [L2=4]"), 
		(10, "coefficients [L2=10]"),
		(1e2, "coefficients [L2=1e2]"), 
		(1e3, "coefficients [L2=1e3]"), 
		(1e5, "coefficients [L2=1e5]")]

	for (l2, label) ∈ KV_LIST
    	print("Running model $(label)")
    	hcoeffs[label] = run_model(;l2_penalty=l2)
	end
end

# ╔═╡ 6d428fe4-78da-11eb-2873-9fd54b4485ea
md"""
```julia

...

Running model coefficients [L2=4]iter: 3 / 1: log likelihood of observed labels: -29098.64917580
iter: 3 / 2: log likelihood of observed labels: -28817.48280975
iter: 3 / 3: log likelihood of observed labels: -28554.20221492
iter: 3 / 4: log likelihood of observed labels: -28306.77144605
iter: 3 / 5: log likelihood of observed labels: -28073.62724612
iter: 3 / 6: log likelihood of observed labels: -27853.41039332
iter: 3 / 7: log likelihood of observed labels: -27644.93882078
iter: 3 / 8: log likelihood of observed labels: -27447.18172819
iter: 3 / 9: log likelihood of observed labels: -27259.23614139
iter: 3 / 10: log likelihood of observed labels: -27080.30692363
iter: 3 / 11: log likelihood of observed labels: -26909.69016432
iter: 3 / 12: log likelihood of observed labels: -26746.75951399
iter: 3 / 13: log likelihood of observed labels: -26590.95496394
iter: 3 / 14: log likelihood of observed labels: -26441.77360865
iter: 3 / 15: log likelihood of observed labels: -26298.76200098
iter: 3 / 20: log likelihood of observed labels: -25663.11897138
iter: 3 / 30: log likelihood of observed labels: -24683.35211346
iter: 3 / 40: log likelihood of observed labels: -23956.08137461
iter: 3 / 50: log likelihood of observed labels: -23390.05514100
iter: 3 / 60: log likelihood of observed labels: -22934.42235545
iter: 3 / 70: log likelihood of observed labels: -22558.31038461
iter: 3 / 80: log likelihood of observed labels: -22241.72295664
iter: 3 / 90: log likelihood of observed labels: -21971.04270712
iter: 3 / 100: log likelihood of observed labels: -21736.62200251
iter: 3 / 200: log likelihood of observed labels: -20412.68742436
iter: 3 / 300: log likelihood of observed labels: -19829.12657619
iter: 3 / 400: log likelihood of observed labels: -19498.14242117
iter: 3 / 500: log likelihood of observed labels: -19285.97423306

Running model coefficients [L2=10]iter: 3 / 1: log likelihood of observed labels: -29098.66002261
iter: 3 / 2: log likelihood of observed labels: -28817.53741305
iter: 3 / 3: log likelihood of observed labels: -28554.32996469
iter: 3 / 4: log likelihood of observed labels: -28306.99840133
iter: 3 / 5: log likelihood of observed labels: -28073.97638005
iter: 3 / 6: log likelihood of observed labels: -27853.90205286
iter: 3 / 7: log likelihood of observed labels: -27645.59110470
iter: 3 / 8: log likelihood of observed labels: -27448.01079546
iter: 3 / 9: log likelihood of observed labels: -27260.25646522
iter: 3 / 10: log likelihood of observed labels: -27081.53150287
iter: 3 / 11: log likelihood of observed labels: -26911.13070122
iter: 3 / 12: log likelihood of observed labels: -26748.42656501
iter: 3 / 13: log likelihood of observed labels: -26592.85806880
iter: 3 / 14: log likelihood of observed labels: -26443.92140137
iter: 3 / 15: log likelihood of observed labels: -26301.16230617
iter: 3 / 20: log likelihood of observed labels: -25666.87619856
iter: 3 / 30: log likelihood of observed labels: -24690.11746503
iter: 3 / 40: log likelihood of observed labels: -23966.03330325
iter: 3 / 50: log likelihood of observed labels: -23403.24288550
iter: 3 / 60: log likelihood of observed labels: -22950.83216858
iter: 3 / 70: log likelihood of observed labels: -22577.89582847
iter: 3 / 80: log likelihood of observed labels: -22264.42031143
iter: 3 / 90: log likelihood of observed labels: -21996.77941010
iter: 3 / 100: log likelihood of observed labels: -21765.32150230
iter: 3 / 200: log likelihood of observed labels: -20467.04068096
iter: 3 / 300: log likelihood of observed labels: -19903.56356097
iter: 3 / 400: log likelihood of observed labels: -19588.90851897
iter: 3 / 500: log likelihood of observed labels: -19390.35674625

...
```
"""

# ╔═╡ af6691e6-78d4-11eb-1d8a-1f920b745781
md"""
#### Compare coefficients

We now compare the coefficients for each of the models that were trained above. We will create a table of features and learned coefficients associated with each of the different L2 penalty values.

Below is a simple helper function that will help us create this table.
"""

# ╔═╡ af2feeb6-78d4-11eb-2873-9fd54b4485ea
df = DataFrame(Dict{Symbol, Vector{String}}(:word => ["(intercept)", important_words...]));

# ╔═╡ d52b25f6-78d5-11eb-36ac-8d6c7b6cdaf9
function add_coeffs_to_df!(df, coeffs, colname)
    insertcols!(df, colname => coeffs)
end

# ╔═╡ 001765f6-78d6-11eb-27e4-55ec5250ae23
for (_l2, label) ∈ KV_LIST
    add_coeffs_to_df!(df, reshape(hcoeffs[label], length(hcoeffs[label])), label)
end

# ╔═╡ 2276fa08-78d6-11eb-3e76-49a4a2f4094f
first(df, 5)

# ╔═╡ 280cdf00-78d6-11eb-37ac-a320a56a628e
md"""
Using the coefficients trained with L2 penalty 0, find the 5 most positive words (with largest positive coefficients). Save them to positive_words. Similarly, find the 5 most negative words (with largest negative coefficients) and save them to negative_words.
"""

# ╔═╡ 323dbeb8-78d6-11eb-08c1-13392c483a14
function top_n(;words=important_words, 
		coeffs=hcoeffs["coefficients [L2=0]"][2:end], key=:positive)
    sort(
      [(w, c) for (w, c) ∈ zip(words, coeffs)],
      by=t -> t[2],
      rev= key == :positive
  )
end

# ╔═╡ 7ecbea02-78d6-11eb-1f51-2729cf921bc5
md"""
**Quiz Question**. Which of the following is **not** listed in either **positive_words** or **negative_words**?"
"""

# ╔═╡ 32217b90-78d6-11eb-05fa-d953b60010ff
positive_words = map(t -> t[1], top_n()[1:5])

# ╔═╡ 3207dc58-78d6-11eb-2b8f-a9b22b5c46fe
negative_words = map(t -> t[1], top_n(;key=:negative)[1:5])

# ╔═╡ 31ef8180-78d6-11eb-218d-ff1d2947d397
md"""
Let us observe the effect of increasing L2 penalty on the 10 words just selected. We provide you with a utility function to plot the coefficient path.
"""

# ╔═╡ 95841904-78d6-11eb-3423-936b80c46605
# TODO plot

# ╔═╡ ab93ec4a-78d6-11eb-1706-15ca470f7cf2
md"""
**Quiz Question**: (True/False) All coefficients consistently get smaller in size as the L2 penalty is increased.
  - True

**Quiz Question**: (True/False) The relative order of coefficients is preserved as the L2 penalty is increased. (For example, if the coefficient for 'cat' was more positive than that for 'dog', this remains true as the L2 penalty increases.)
  - True
"""

# ╔═╡ 6bd80650-780f-11eb-20f9-8f99728a72f3
md"""
### Measuring accuracy

We will now measure the classification accuracy of the model. Recall from the lecture that the classification accuracy can be computed as follows:

```math
\mbox{accuracy} = \frac{{\mbox{\# correctly classified data points}}}{\mbox{\# total data points}}
```

Recall from lecture that that the class prediction is calculated using
```math
\hat{y}_i = 
\left\{
\begin{array}{ll}
      +1 & h(\mathbf{x}_i)^T\mathbf{w} > 0 \\
      -1 & h(\mathbf{x}_i)^T\mathbf{w} \leq 0 \\
\end{array} 
\right.
```

**Note**: It is important to know that the model prediction code doesn't change even with the addition of an L2 penalty. The only thing that changes is the estimated coefficients used in this prediction.

"""

# ╔═╡ dcef9dea-78d6-11eb-267b-236c60163a33
function get_classif_accuracy(feature_matrix, sentiment, coeffs)
	scores = feature_matrix * coeffs
  	preds = ifelse.(scores .> 0., 1, -1)
	num_correct = sum(preds .== sentiment)
    return num_correct / size(feature_matrix)[1]  ## == accuracy
end

# ╔═╡ 30861830-78d7-11eb-0389-5978263881b6
md"""
Below, we compare the accuracy on the training data and validation data for all the models that were trained in this assignment. We first calculate the accuracy values and then build a simple report summarizing the performance for the various models.
"""

# ╔═╡ 5b0e8fd0-78d7-11eb-13d1-8dafac6a1d6a
begin
	train_accuracy, validation_accuracy = Dict(), Dict()

	for (l2, label) ∈ KV_LIST
    	train_accuracy[l2] = get_classif_accuracy(feature_matrix_train, 
			sentiment_train, hcoeffs[label])
    	validation_accuracy[l2] = get_classif_accuracy(feature_matrix_valid, 
			sentiment_valid, hcoeffs[label])
	end
end

# ╔═╡ 5af4099c-78d7-11eb-12f9-1f5e36d0706d
begin
  with_terminal() do
	# Build a simple report
	for key ∈ keys(validation_accuracy) |> collect |> sort
    	@printf("L2 penalty = %g", key)
    	@printf(" train accuracy = %1.3f, validation_accuracy = %1.3f\n", 
				train_accuracy[key], validation_accuracy[key])
        println("----------------------------------------------------")
	end
  end
end

# ╔═╡ 8cee74b0-78d8-11eb-3acd-9bc21f53815e
# TODO Plot

# ╔═╡ 5ad9fbba-78d7-11eb-0209-6533fe570748
md"""
* **Quiz Question**: Which model (L2 = 0, 4, 10, 100, 1e3, 1e5) has the **highest** accuracy on the **training** data?
  - model with L2 = 0 (first model)
  
* **Quiz Question**: Which model (L2 = 0, 4, 10, 100, 1e3, 1e5) has the **highest** accuracy on the **validation** data?
  - model with L2 = 10 (third model)
  
* **Quiz Question**: Does the **highest** accuracy on the **training** data imply that the model is the best one?
  - No (it can indicate that the model is overfitting the training data, and therefore won't generalize well, unless...)
"""

# ╔═╡ Cell order:
# ╟─baa8ca1c-77d8-11eb-2e26-3b2445acbfa1
# ╠═33bd93ec-77d9-11eb-2da8-87271f8b30c7
# ╟─3ad4770e-77d9-11eb-3a93-addf3e427025
# ╠═6e264178-77d9-11eb-33c1-19462dde66e2
# ╠═75f2e820-77de-11eb-19eb-a5a5977b5257
# ╠═36f2c616-77de-11eb-03b4-3b92a532949a
# ╠═6d4ef592-77d9-11eb-3844-1bea7dc71094
# ╠═d7b9c812-77de-11eb-0ebb-6bb75de9725b
# ╟─6e09ca34-77d9-11eb-1a99-6578f9141004
# ╠═6ded8978-77d9-11eb-3d94-e7507a85197e
# ╟─6dd12c56-77d9-11eb-0350-1dfc08d17dd6
# ╠═6db87918-77d9-11eb-2082-edecdf8b7576
# ╠═6d69b148-77d9-11eb-04f5-034603897c31
# ╟─6d316f86-77d9-11eb-107b-1bf5708655d5
# ╠═0f6bb770-77df-11eb-0f1f-4dddbd7e2498
# ╟─734ed4d4-77df-11eb-3dab-e5c10b4cda62
# ╠═0f528d0e-77df-11eb-1415-2fb0e9d6792d
# ╠═0f362fec-77df-11eb-02a6-210a58c2839b
# ╟─0f1c6b16-77df-11eb-2197-93c4a82cc135
# ╠═0057a50e-77e0-11eb-29d6-85c2dac67743
# ╠═51a94e8e-77e3-11eb-04c8-d1389b1f5856
# ╠═0f03e474-77df-11eb-2791-f165fd011737
# ╠═0ee68550-77df-11eb-0a6e-c5bf5e3fd9ac
# ╠═0ecc0054-77df-11eb-21a4-3f677548ecf9
# ╟─0eb50d9a-77df-11eb-0b7d-557ab3ea2c81
# ╟─d39040fe-78d0-11eb-236d-15b506adaffd
# ╠═1c52f3b6-78d1-11eb-3077-bf6c471251c8
# ╠═005dc370-78d1-11eb-0fc0-039f1017af8e
# ╠═004d22cc-78d1-11eb-10ce-09eb125a96b3
# ╟─682a78ba-78d1-11eb-1706-15ca470f7cf2
# ╠═0e383a04-77df-11eb-2ec9-c5aeffc4b131
# ╠═7ad4c752-78d1-11eb-267b-236c60163a33
# ╠═9f29e2b8-78d1-11eb-0389-5978263881b6
# ╠═b093b9ac-78d1-11eb-3a2e-ffb6e1102b1f
# ╟─fd23d180-77e6-11eb-2091-af1e84d8fbcf
# ╠═fd120ad4-77e6-11eb-15ea-bb7d7ea35587
# ╟─fcf1dd2e-77e6-11eb-0619-e7fd2a622877
# ╟─d6f4542a-78d2-11eb-0209-6533fe570748
# ╠═d6976230-78d2-11eb-1fb7-dd1259c9b9c8
# ╟─4870ab12-78d3-11eb-12f9-1f5e36d0706d
# ╟─665516c2-78d3-11eb-13d1-8dafac6a1d6a
# ╠═70a490b2-78d3-11eb-3acd-9bc21f53815e
# ╟─0cea789c-78d4-11eb-36c4-c794c5e1a816
# ╟─1a7b28f8-78d4-11eb-3906-5ba3566b9ae9
# ╠═970fba24-77ed-11eb-1a3e-e769d6049078
# ╟─972e0c72-77ed-11eb-0ae7-c127694faf89
# ╠═afa2bff4-78d4-11eb-10cc-0deed0adb371
# ╠═af858fea-78d4-11eb-1499-3558c5ea680c
# ╟─6d428fe4-78da-11eb-2873-9fd54b4485ea
# ╟─af6691e6-78d4-11eb-1d8a-1f920b745781
# ╠═af2feeb6-78d4-11eb-2873-9fd54b4485ea
# ╠═d52b25f6-78d5-11eb-36ac-8d6c7b6cdaf9
# ╠═001765f6-78d6-11eb-27e4-55ec5250ae23
# ╠═2276fa08-78d6-11eb-3e76-49a4a2f4094f
# ╟─280cdf00-78d6-11eb-37ac-a320a56a628e
# ╠═323dbeb8-78d6-11eb-08c1-13392c483a14
# ╟─7ecbea02-78d6-11eb-1f51-2729cf921bc5
# ╠═32217b90-78d6-11eb-05fa-d953b60010ff
# ╠═3207dc58-78d6-11eb-2b8f-a9b22b5c46fe
# ╟─31ef8180-78d6-11eb-218d-ff1d2947d397
# ╠═95841904-78d6-11eb-3423-936b80c46605
# ╟─ab93ec4a-78d6-11eb-1706-15ca470f7cf2
# ╟─6bd80650-780f-11eb-20f9-8f99728a72f3
# ╠═dcef9dea-78d6-11eb-267b-236c60163a33
# ╟─30861830-78d7-11eb-0389-5978263881b6
# ╠═5b0e8fd0-78d7-11eb-13d1-8dafac6a1d6a
# ╠═5af4099c-78d7-11eb-12f9-1f5e36d0706d
# ╠═8cee74b0-78d8-11eb-3acd-9bc21f53815e
# ╟─5ad9fbba-78d7-11eb-0209-6533fe570748
