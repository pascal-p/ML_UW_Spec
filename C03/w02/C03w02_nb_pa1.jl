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
include("./C03w01_text_proc.jl");

# ╔═╡ baa8ca1c-77d8-11eb-2e26-3b2445acbfa1
md"""
## Implementing logistic regression from scratch

The goal of this notebook is to implement your own logistic regression classifier. We will:

  - Extract features from Amazon product reviews.
  - Convert an DataFrame into a Julia array.
  - Implement the link function for logistic regression.
  - Write a function to compute the derivative of the log likelihood function with respect to a single coefficient.
  - Implement gradient ascent.
  - Given a set of coefficients, predict sentiments.
  - Compute classification accuracy for the logistic regression model.
 
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

# ╔═╡ 6d9e0568-77d9-11eb-1369-bb6990e1e4a7
# num. of positive reviews, num. of negative reviews
(num_pos=sum(products.sentiment .== 1), num_neg=sum(products.sentiment .== -1))

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
# products[!, :review_clean] = remove_punctuation.(products[:, :review]);
products[!, :review] = remove_punctuation.(products[:, :review]);

# ╔═╡ 0f362fec-77df-11eb-02a6-210a58c2839b
names(products)

# ╔═╡ 0f1c6b16-77df-11eb-2197-93c4a82cc135
md"""
###### Step 2.

For each word in `important_words`, we compute a count for the number of times the word occurs in the review. We will store this count in a separate column (one for each word). The result of this feature processing is a single column for each word in `important_words` which keeps a count of the number of times the respective word occurs in the review text.

"""

# ╔═╡ 0057a50e-77e0-11eb-29d6-85c2dac67743
function wc(text::String, word::String)
 	# length(filter(w -> w == word, split(text)))
	length(findall(word, text))
end

# ╔═╡ 51a94e8e-77e3-11eb-04c8-d1389b1f5856
function addcols!(df, words)
	for word ∈ words
		df[!, word] = wc.(df.review_clean, word)
	end
	
	#for word ∈ words
	#	transform!(df, :review_clean => r -> wc.(r, word) => word)
	#end
end

# ╔═╡ 0f03e474-77df-11eb-2791-f165fd011737
@time addcols!(products, important_words)
# 135.474563 seconds (2.43 G allocations: 181.104 GiB, 9.77% gc time)
# transform!  5.6s
# df[!, wrod] 5.5s

# ╔═╡ 0ee68550-77df-11eb-0a6e-c5bf5e3fd9ac
length(names(products))

# ╔═╡ 0ecc0054-77df-11eb-21a4-3f677548ecf9
products[!, :perfect], length(products[!, :perfect])

# ╔═╡ 0eb50d9a-77df-11eb-0b7d-557ab3ea2c81
md"""
Now, write some code to compute the number of product reviews that contain the word perfect.
"""

# ╔═╡ 0e9884fe-77df-11eb-3199-13ab2fbdcbc8
sum(products.perfect .>= 1)

# ╔═╡ 0e815eaa-77df-11eb-3422-515926dab50d
md"""
**Quiz Question**. How many reviews contain the word perfect?
  - cf. above cell.
"""

# ╔═╡ 0e4f6222-77df-11eb-3c39-239382844c12
md"""
#### Using Julia array
"""

# ╔═╡ 0e383a04-77df-11eb-2ec9-c5aeffc4b131
function get_data(df::DF, features, output) where {DF <: Union{DataFrame, SubDataFrame}}
	s_features = [Symbol(f) for f ∈ features]
	df[:, :intercept] .= 1.0 
	s_features = [:intercept, s_features...]
	X_matrix = convert(Matrix, select(df, s_features)) 
	y = df[!, output]                                
	
	(X_matrix, y)
end

# ╔═╡ 0e1d5f86-77df-11eb-12c9-75b0b884ad25
@time feature_matrix, sentiment = get_data(products, important_words, :sentiment); 

# ╔═╡ 0e03942c-77df-11eb-1bde-4f7b6739a3ee
size(feature_matrix)

# ╔═╡ 0deb769c-77df-11eb-2b53-7b73905353cd
md"""
**Quiz Question**: How many features are there in the feature_matrix?
"""

# ╔═╡ 0dd0bc6c-77df-11eb-2df1-09ca3d5aaafc
size(feature_matrix)[2]

# ╔═╡ fd5a8f7c-77e6-11eb-34dd-11ba6b2d7409
md"""
**Quiz Question**: Assuming that the intercept is present, how does the number of features in feature_matrix relate to the number of features in the logistic regression model?
Let x = [number of features in feature_matrix] and y = [number of features in logistic regression model].

  - [ ] y = x - 1
  - [x] y = x
  - [ ] y = x + 1
  - [ ] None of the above

"""

# ╔═╡ fd419c9c-77e6-11eb-2925-0d832a3fe2ca
sentiment

# ╔═╡ fd23d180-77e6-11eb-2091-af1e84d8fbcf
md"""
####  Estimating conditional probability with link function

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
**Aside**. How the link function works with matrix algebra

Since the word counts are stored as columns in **feature_matrix**, each $i$-th row of the matrix corresponds to the feature vector $h(\mathbf{x}_i)$:


```math
[feature\_matrix] = 
\left[
	\begin{array}{c}
		h(\mathbf{x}_1)^T \\
		h(\mathbf{x}_2)^T \\
		\vdots \\
		h(\mathbf{x}_N)^T
	\end{array}
\right] = \left[
	\begin{array}{cccc}
	h_0(\mathbf{x}_1) & h_1(\mathbf{x}_1) & \cdots & h_D(\mathbf{x}_1) \\
	h_0(\mathbf{x}_2) & h_1(\mathbf{x}_2) & \cdots & h_D(\mathbf{x}_2) \\
	\vdots & \vdots & \ddots & \vdots \\
	h_0(\mathbf{x}_N) & h_1(\mathbf{x}_N) & \cdots & h_D(\mathbf{x}_N)
\end{array}
\right]
```

By the rules of matrix multiplication, the score vector containing elements $\mathbf{w}^T h(\mathbf{x}_i)$ is obtained by multiplying **feature_matrix** and the coefficient vector $\mathbf{w}$.

```math
[score] =
[feature\_matrix]\mathbf{w} =
\left[
\begin{array}{c}
h(\mathbf{x}_1)^T \\
h(\mathbf{x}_2)^T \\
\vdots \\
h(\mathbf{x}_N)^T
\end{array}
\right]
\mathbf{w}
= \left[
\begin{array}{c}
h(\mathbf{x}_1)^T\mathbf{w} \\
h(\mathbf{x}_2)^T\mathbf{w} \\
\vdots \\
h(\mathbf{x}_N)^T\mathbf{w}
\end{array}
\right]
= \left[
\begin{array}{c}
\mathbf{w}^T h(\mathbf{x}_1) \\
\mathbf{w}^T h(\mathbf{x}_2) \\
\vdots \\
\mathbf{w}^T h(\mathbf{x}_N)
\end{array}
\right]
```
"""

# ╔═╡ bb677796-77e7-11eb-2baf-ffb0ee41fff6
md"""
**Checkpoint**

Just to make sure we are on the right track, we have provided a few examples. If our predict_probability function is implemented correctly, then the following should pass

"""

# ╔═╡ bb4f23d0-77e7-11eb-1f2d-efd11a1d5f66
begin
	d_feature_matrix = [1. 2. 3.; 1. -1. -1]
	d_coeffs = [1.; 3.; -1]

	# c_scores = d_feature_matrix .* d_coeffs
	c_scores = sum(d_feature_matrix .* d_coeffs', dims=2)
	c_preds  = [1. ./ (1. .+ exp.(-c_scores[1])); 1. / (1. .+ exp.(-c_scores[2]))]

	ϵ = 1e-7
	@test all(t -> abs(t[1] - t[2]) ≤ ϵ, 
		zip(predict_probability(d_feature_matrix, d_coeffs), c_preds))
end

# ╔═╡ 9744c78c-77ed-11eb-3bf6-fdc3d8cf5281
md"""
#### Compute derivative of log likelihood with respect to a single coefficient

Recall from lecture:

```math
\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right)
```

We will now write a function that computes the derivative of log likelihood with respect to a single coefficient $w_j$. The function accepts two arguments:
  - `errors` vector containing $\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})$ for all $i$.
  - `feature` vector containing $h_j(\mathbf{x}_i)$  for all $i$. 


"""

# ╔═╡ 2cddfe30-77ee-11eb-389c-130f0af51f70
feature_derivative(errors::Matrix, feature::Vector) = dot(errors, feature)

# ╔═╡ 2cbeab98-77ee-11eb-2108-7596888619c2
md"""
In the main lecture, our focus was on the likelihood.  In the advanced optional video, however, we introduced a transformation of this likelihood *called the log likelihood* that simplifies the derivation of the gradient and is more numerically stable.  Due to its numerical stability, we will use the log likelihood instead of the likelihood to assess the algorithm.

The log likelihood is computed using the following formula (see the advanced optional video if you are curious about the derivation of this equation):

$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big)$$
 
"""

# ╔═╡ 2ca74a52-77ee-11eb-0f50-fd3c15702be1
function log_likelihood(feature_matrix::Matrix, sentiment, coeffs)
	indic = sentiment .== 1
	scores = feature_matrix * coeffs
	log_exp = log.(1. .+ exp.(-scores))
	
	mask = isinf.(log_exp)
	log_exp[mask] = -scores[mask]
	
	sum((indic .- 1) .* scores .- log_exp)
end 

# ╔═╡ 2c8d9e7c-77ee-11eb-3897-c5bd68dbea53
md"""
**Checkpoint**

Just to make sure we are on the same page, the following code block should pass.
"""

# ╔═╡ 889c2692-77ef-11eb-057c-47b467b71f21
begin
	# re-using d_feature_matrix, d_coeffs above
	d_sentiment = [-1; 1]

	c_indic  = [-1 == +1; 1 == +1] # [Int(-1 == +1); Int(1 == +1)]
	# c_scores (as above)
	
	c_1st_term  = [(c_indic[1] - 1) * c_scores[1]; (c_indic[2] - 1) * c_scores[2]]
	c_2nd_term = [log(1. + exp(-c_scores[1])); log(1. + exp(-c_scores[2]))]

	c_ll = sum([c_1st_term[1] .- c_2nd_term[1]; c_1st_term[2] .- c_2nd_term[2]]) 

	@test abs(log_likelihood(d_feature_matrix, d_sentiment, d_coeffs) - c_ll) ≤ ϵ
end

# ╔═╡ 972e0c72-77ed-11eb-0ae7-c127694faf89
md"""
#### Taking gradient steps

Now we are ready to implement our own logistic regression. All we have to do is to write a gradient ascent function that takes gradient steps towards the optimum. 
"""

# ╔═╡ 970fba24-77ed-11eb-1a3e-e769d6049078
function logistic_regression(feature_matrix::Matrix, sentiment, init_coeffs; 
		η=1e-7, max_iter=200)
	coeffs = copy(init_coeffs)
	# println("coeffs: $(size(coeffs))")
	
	for ix ∈ 1:max_iter
		preds = predict_probability(feature_matrix, coeffs)
		indic = sentiment .== 1
		errors = indic .- preds
		# println(size(preds), size(indic), size(errors))
		
		for jx ∈ 1:length(coeffs)
			# println("==> $(size(feature_matrix[:, jx]))")
			∂c = feature_derivative(errors, feature_matrix[:, jx])
			# println("coeffs: $(size(coeffs[jx])), ∂c: $(size(∂c))")
			coeffs[jx] += η * ∂c
		end
		
		if ix ≤ 15 || (ix ≤ 100 && ix % 10 == 0) || (ix ≤ 1000 && ix % 100 == 0) ||
			(ix ≤ 10000 && ix % 1000 == 0) || ix % 10000 == 0
			lp = log_likelihood(feature_matrix, sentiment, coeffs)
			@printf("iter: %d / %d: log likelihood of observed labels: %.8f\n", 
				ceil(Int, log10(max_iter)), ix, lp)
		end
 	end
	
	coeffs
end

# ╔═╡ 96f4d234-77ed-11eb-0e9d-8b4483b9d020
@time coeffs = logistic_regression(feature_matrix, sentiment, 
	zeros(Float64, (194, 1)); max_iter=301);

# ╔═╡ 656b8c04-780c-11eb-16ef-2d0ab940e7b7
size(feature_matrix)

# ╔═╡ a01911d6-7808-11eb-34e1-9dde4224fca8
md"""
**Quiz Question:** As each iteration of gradient ascent passes, does the log likelihood increase or decrease?
  - increase
"""

# ╔═╡ a000fdf0-7808-11eb-32af-fba57700380e
md"""
### Predicting sentiments

Recall from lecture that class predictions for a data point $\mathbf{x}$ can be computed from the coefficients $\mathbf{w}$ using the following formula:

```math
\hat{y}_i = 
\left\{
\begin{array}{ll}
      +1 & \mathbf{x}_i^T\mathbf{w} > 0 \\
      -1 & \mathbf{x}_i^T\mathbf{w} \leq 0 \\
\end{array} 
\right.
```

Now, we will write some code to compute class predictions. We will do this in two steps:
  1. First compute the **scores** using **feature_matrix** and **coefficients** using a dot product.
  2.  Using the formula above, compute the class predictions from the scores.

Step 1 can be implemented as follows:
"""

# ╔═╡ 9fe93efc-7808-11eb-3755-11a93bd3374a
function calc_preds(feature_matrix::Matrix, coeffs)
	scores = feature_matrix * coeffs
	ifelse.(scores .> 0., 1, -1)
end

# ╔═╡ 9fd0d0b0-7808-11eb-2f66-c5815f2f4929
preds = calc_preds(feature_matrix, coeffs);

# ╔═╡ 9fb85b86-7808-11eb-1af6-0f648740b822
md"""
**Quiz Question:** How many reviews were predicted to have positive sentiment?
"""

# ╔═╡ 380fef64-780c-11eb-3676-7d2fa5d2f726
begin
	n_pos, n_neg = length(preds[preds .== 1]), length(preds[preds .== -1])
	@assert n_pos + n_neg == length(preds)

	(pred_pos=n_pos, pred_neg=n_neg)
end

# ╔═╡ 6bd80650-780f-11eb-20f9-8f99728a72f3
md"""
### Measuring accuracy

We will now measure the classification accuracy of the model. Recall from the lecture that the classification accuracy can be computed as follows:

```math
\mbox{accuracy} = \frac{{\mbox{\# correctly classified data points}}}{\mbox{\# total data points}}
```
"""

# ╔═╡ c7ff6cc0-780f-11eb-3ac5-ffd1fd251698
t = (num_mistakes=sum(sentiment .- preds .!= 0), total_data_points=size(products)[1])

# ╔═╡ 7e0f62a4-7810-11eb-34d9-85a2f2de4296
md"""
**Quiz Question**: What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)
"""

# ╔═╡ c7e77f86-780f-11eb-0511-5394adb6b0dd
begin
	accuracy = (t.total_data_points - t.num_mistakes) / t.total_data_points;
	round(accuracy, digits=2)
end

# ╔═╡ 8b92d55a-7810-11eb-2495-0112a869bcc9
md"""
### Which words contribute most to positive & negative sentiments?

Recall that in Module 2 assignment, we were able to compute the "**most positive words**". These are words that correspond most strongly with positive reviews. In order to do this, we will first do the following:

  - Treat each coefficient as a tuple, i.e. (**word**, **coefficient_value**).

  - Sort all the (**word**, **coefficient_value**) tuples by **coefficient_value** in descending order.
"""

# ╔═╡ 8c6e485a-7811-11eb-3fe7-19876cc85ee4
md"""
##### Ten "most positive" words

Now, we compute the 10 words that have the most positive coefficient values. These words are associated with positive sentiment.

"""

# ╔═╡ c7cacf90-780f-11eb-21ac-37de331bfd0d
function top_n(;words=important_words, coeffs=coeffs[2:end], key=:positive)
  	sort(
    	[(w, c) for (w, c) ∈ zip(words, coeffs)], 
    	by=t -> t[2], 
    	rev= key == :positive
	)
end

# ╔═╡ 28b5ed9a-7811-11eb-2e7a-e9ee7b3c7db2
md"""
**Quiz Question**: Which word is **NOT** present in the top 10 "most positive" words?
"""

# ╔═╡ 42107bfc-7811-11eb-35f7-f52a103f7be6
map(t -> t[1], top_n()[1:10])

# ╔═╡ 7321412c-7811-11eb-14eb-f529581256f1
md"""
  - **cheap** is NOT in previous list.
"""

# ╔═╡ 86669250-7811-11eb-1938-5dedabd25ad7
md"""
##### Ten "most negative" words

Next, we repeat this exercise on the 10 most negative words.  That is, we compute the 10 words that have the most negative coefficient values. These words are associated with negative sentiment.
"""

# ╔═╡ 864d674e-7811-11eb-0071-85e6c429d503
map(t -> t[1], top_n(;key=:negative)[1:10])

# ╔═╡ 862e1b1e-7811-11eb-1212-25bce5527b8b
md"""
  - **need** is NOT in previous list.
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
# ╠═6d9e0568-77d9-11eb-1369-bb6990e1e4a7
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
# ╠═0e9884fe-77df-11eb-3199-13ab2fbdcbc8
# ╟─0e815eaa-77df-11eb-3422-515926dab50d
# ╟─0e4f6222-77df-11eb-3c39-239382844c12
# ╠═0e383a04-77df-11eb-2ec9-c5aeffc4b131
# ╠═0e1d5f86-77df-11eb-12c9-75b0b884ad25
# ╠═0e03942c-77df-11eb-1bde-4f7b6739a3ee
# ╟─0deb769c-77df-11eb-2b53-7b73905353cd
# ╠═0dd0bc6c-77df-11eb-2df1-09ca3d5aaafc
# ╟─fd5a8f7c-77e6-11eb-34dd-11ba6b2d7409
# ╠═fd419c9c-77e6-11eb-2925-0d832a3fe2ca
# ╟─fd23d180-77e6-11eb-2091-af1e84d8fbcf
# ╠═fd120ad4-77e6-11eb-15ea-bb7d7ea35587
# ╟─fcf1dd2e-77e6-11eb-0619-e7fd2a622877
# ╟─bb677796-77e7-11eb-2baf-ffb0ee41fff6
# ╠═bb4f23d0-77e7-11eb-1f2d-efd11a1d5f66
# ╟─9744c78c-77ed-11eb-3bf6-fdc3d8cf5281
# ╠═2cddfe30-77ee-11eb-389c-130f0af51f70
# ╟─2cbeab98-77ee-11eb-2108-7596888619c2
# ╠═2ca74a52-77ee-11eb-0f50-fd3c15702be1
# ╟─2c8d9e7c-77ee-11eb-3897-c5bd68dbea53
# ╠═889c2692-77ef-11eb-057c-47b467b71f21
# ╟─972e0c72-77ed-11eb-0ae7-c127694faf89
# ╠═970fba24-77ed-11eb-1a3e-e769d6049078
# ╠═96f4d234-77ed-11eb-0e9d-8b4483b9d020
# ╠═656b8c04-780c-11eb-16ef-2d0ab940e7b7
# ╟─a01911d6-7808-11eb-34e1-9dde4224fca8
# ╟─a000fdf0-7808-11eb-32af-fba57700380e
# ╠═9fe93efc-7808-11eb-3755-11a93bd3374a
# ╠═9fd0d0b0-7808-11eb-2f66-c5815f2f4929
# ╟─9fb85b86-7808-11eb-1af6-0f648740b822
# ╠═380fef64-780c-11eb-3676-7d2fa5d2f726
# ╟─6bd80650-780f-11eb-20f9-8f99728a72f3
# ╠═c7ff6cc0-780f-11eb-3ac5-ffd1fd251698
# ╟─7e0f62a4-7810-11eb-34d9-85a2f2de4296
# ╠═c7e77f86-780f-11eb-0511-5394adb6b0dd
# ╟─8b92d55a-7810-11eb-2495-0112a869bcc9
# ╟─8c6e485a-7811-11eb-3fe7-19876cc85ee4
# ╠═c7cacf90-780f-11eb-21ac-37de331bfd0d
# ╟─28b5ed9a-7811-11eb-2e7a-e9ee7b3c7db2
# ╠═42107bfc-7811-11eb-35f7-f52a103f7be6
# ╟─7321412c-7811-11eb-14eb-f529581256f1
# ╟─86669250-7811-11eb-1938-5dedabd25ad7
# ╠═864d674e-7811-11eb-0071-85e6c429d503
# ╟─862e1b1e-7811-11eb-1212-25bce5527b8b
