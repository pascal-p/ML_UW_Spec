### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ eb97df10-76f1-11eb-21da-378e76583695
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
	
	using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	using Test
	using Printf
	# using Plots
end


# ╔═╡ 299e3eaa-7709-11eb-2b3e-13dcfbd9f450
begin
	include("./utils.jl");
	include("./text_proc.jl");
end

# ╔═╡ 9cf87de4-76f1-11eb-221a-8fbc8dad79fd
md"""
## C03/w01: Predicting sentiment from product reviews


The goal of this first notebook is to explore logistic regression and feature engineering with Julia and MLJ.

In this notebook we will use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative.

  -  Use DataFrames to do some feature engineering
  - Train a logistic regression model to predict the sentiment of product reviews.
  -  Inspect the weights (coefficients) of a trained logistic regression model.
  - Make a prediction (both class and probability) of sentiment for a new product review.
  -  Given the logistic regression weights, predictors and ground truth labels, write a function to compute the **accuracy** of the model.
  -  Inspect the coefficients of the logistic regression model and interpret their meanings.
  -  Compare multiple logistic regression models.
"""

# ╔═╡ fea86e6c-76f1-11eb-0188-a1d8b7bcf47b
md"""
### Data Preparation

We will use a dataset consisting of baby product reviews on Amazon.com. Let us load it and see a preview of what it looks like.
"""

# ╔═╡ 17de9582-76f2-11eb-03cf-43ad5b07d120
begin
	products = train = CSV.File("../../ML_UW_Spec/C03/data/amazon_baby.csv"; 
		header=true) |> DataFrame;

	size(products)
end

# ╔═╡ f29e95dc-7d50-11eb-1ecb-51f578ef29f0
md"""
###### Deal with missing values

  1. Filling missing review with empty string
  2. drop other missing rows 
"""

# ╔═╡ 01417a14-7d51-11eb-394e-a36a82d4a40a
replace!(products.review, missing =>"");

# ╔═╡ ade2915c-7709-11eb-2fce-45999f1e9ffa
begin
	## drop missing value if any...
	before_rm_na = size(products)
	dropmissing!(products);
	before_rm_na, size(products)
end

# ╔═╡ 67034e64-76f2-11eb-2a25-c53adf9adde6
md"""
### Build the word count vector for each review

Let us explore a specific example of a baby product.
"""

# ╔═╡ 7680d44c-76f2-11eb-1ffe-f92fdd5e42b7
products[270, :]

# ╔═╡ db3bbb6e-76f6-11eb-09c1-e7f0392c634d
products[70, :review]

# ╔═╡ 76610114-76f2-11eb-3e22-71a763599e47
md"""
Now, we will perform 2 simple data transformations:

  - Remove punctuation (using a regular expression) and
  - Transform the reviews into word-counts.

Aside. In this notebook, we remove all punctuations for the sake of simplicity. A smarter approach to punctuations would preserve phrases such as "I'd", "would've", "hadn't" and so forth.

"""

# ╔═╡ 762adc54-76f2-11eb-259d-8f6cae34ef20
remove_punctuation(" Voici un example; ponctuation: elimine? Vraiment?? Oui! *****")

# ╔═╡ 75e24a66-76f2-11eb-2711-f53a6c04fb99
products[!, :review] = remove_punctuation.(products[:, :review]);

# ╔═╡ 75bfd17a-76f2-11eb-3285-31b3eaa96a62
products[70, :review]

# ╔═╡ 3b78e5bc-7707-11eb-1e32-2116b3156727
remove_punctuation("   Voici un example; ponctuation: elimine. Vraiment?? Oui! *****")   |> word_count

# ╔═╡ b398fef4-7706-11eb-2b36-17f85f356a3c
first(products, 3)

# ╔═╡ 42d86fd8-7878-11eb-0e98-c73dd498e285
md"""
#### Clean up review
"""

# ╔═╡ 861170cc-7877-11eb-00c7-0f50fe4ceeea
## clean review
begin
	products[!, :review_clean] = cleanup.(products.review);
	select!(products, Not(:review));
	size(products)
end

# ╔═╡ d4844b50-770b-11eb-221c-0f281a8d8eec
md"""
#### Extract sentiment

We will ignore all reviews with `rating=3`, since they tend to have a neutral sentiment.

Now, we will assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative.
For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.

"""

# ╔═╡ f40d002a-770b-11eb-3d91-8916193cf7a3
begin
	filter!(r -> r.rating != 3, products);
	products.sentiment = ifelse.(products.rating .> 3, 1 ,-1);
	size(products)
end

# ╔═╡ ce050256-7aee-11eb-1474-b96dcad21315
md"""
### MLJ Data Preparation
"""

# ╔═╡ 5fcd5f68-7715-11eb-3667-7d4960547cff
md"""
#### CountVectorizer 
"""

# ╔═╡ 187d5a38-7875-11eb-01c1-bbc44e1d15c0
token_ix = gen_vocab.(products.review_clean) |>
   vtext -> reduce(vcat, vtext) |>
   unique |>
   words -> filter(w -> length(w) > 2, words) |>
   vocab -> Dict{String, Int}(w => ix for (ix, w) in enumerate(vocab))
## 58836 unique words -> 58584

# ╔═╡ 0f7839a6-787e-11eb-2887-f34311d11b56
length(token_ix)

# ╔═╡ a6d8c4c2-771c-11eb-30bc-332df04ade43
transform!(products, :review_clean => h_encode => :word_count);

# ╔═╡ 4dda42ea-774b-11eb-2136-b345e017b297
names(products)

# ╔═╡ 9c2dd014-7751-11eb-1ecb-51f578ef29f0
first(products, 3)

# ╔═╡ e265712c-7aee-11eb-28c8-9952ca7fbc46
## before coercion
schema(products)

# ╔═╡ 375e9688-7af2-11eb-08fe-efb6d6b91863
begin
	## Note we only use :word_count as a feature for our Logistic Regr. classifier
	## ST = Scientific Type
	
	const Features_ST = Dict{Symbol, DataType}(
		:word_count => Continuous,             ## num. of unique words...
	# 	:word_count => Count,
	)
	
	Target_ST = Dict{Symbol, DataType}(
		:sentiment => Multiclass{2}
	)
end

# ╔═╡ ef617988-7d59-11eb-2f81-69e21d6c7ffa
elscitype(products.word_count)

# ╔═╡ 69aa0492-7d5c-11eb-2f81-69e21d6c7ffa
scitype(products.word_count)

# ╔═╡ 27f66dec-7d4f-11eb-29a5-d79c586de1d8
## Coerce
begin
	coerce_map = [Features_ST..., Target_ST...]
	coerce!(products, coerce_map...);
	
	# then
	# coerce!(products, autotype(products, :discrete_to_continuous))
	schema(products)
end

# ╔═╡ eced8604-7d44-11eb-246e-a77066a91eb1
# reduce(vcat, products[!, :name]) |> unique |> length # == 30628
# name => Multiclass{N} ?

# ╔═╡ 7e24bb38-770a-11eb-03ef-590069fda67c
md"""
#### Split data into train and test sets
"""

# ╔═╡ b3580318-7706-11eb-0b23-6b817d62e154
begin
	train_data, test_data = train_test_split(products; split=0.8, seed=42)
	size(train_data), size(test_data)
end

# ╔═╡ 909e2708-770b-11eb-0955-6f3c7c6d19ec
names(train_data)

# ╔═╡ b33d33f8-7706-11eb-1b83-130c4fb81eb2
md"""
### Train a sentiment classifier with logistic regression


We will now use logistic regression to create a sentiment classifier on the training data. This model will use the column word_count as a feature and the column sentiment as the target. 
"""

# ╔═╡ a572e232-770a-11eb-12ce-d5cf2e70d878
@load LogisticClassifier pkg=MLJLinearModels

# ╔═╡ e3809494-7af6-11eb-33b9-4d503e9aa1fc
function data_prep_(df::DF; features=Features_ST,
	target=Target_ST) where {DF <: Union{DataFrame, SubDataFrame}}

 	y_target = collect(keys(target))[1]
    x_features = collect(keys(features))
	
	X = select(df, x_features)
	y = select(df, y_target) 
	y = y[:, y_target]
	
   	return (X, y)
end

# ╔═╡ 29716c86-7d53-11eb-1de9-7ba4e5be7112
begin
	Xₜ, yₜ = data_prep_(train_data)
	schema(Xₜ)
end

# ╔═╡ be5a7876-7d56-11eb-145f-45c61470f7cc
first(Xₜ, 3)

# ╔═╡ 46c01a6c-7d53-11eb-0e1b-cbc471d0a34e
typeof(Xₜ), typeof(yₜ)

# ╔═╡ a556fdf6-770a-11eb-18c3-3fc37fe4fe32
function run_logistic_regression(X, y)
	mach = machine(MLJLinearModels.LogisticClassifier(), X, y)
	fit!(mach) # , rows=1:nrows, verbosity=1, force=false)
	mach
end

# ╔═╡ b30dea8a-7706-11eb-266b-eb9027195a0b
begin
	sentiment_mdl = run_logistic_regression(Xₜ, yₜ);
	fitted_params(sentiment_mdl) # report(sentiment_mdl)
	# ref: https://github.com/alan-turing-institute/MLJ.jl/issues/492
end

# ╔═╡ fc20085e-7d53-11eb-2f81-69e21d6c7ffa


# ╔═╡ Cell order:
# ╟─9cf87de4-76f1-11eb-221a-8fbc8dad79fd
# ╠═eb97df10-76f1-11eb-21da-378e76583695
# ╟─fea86e6c-76f1-11eb-0188-a1d8b7bcf47b
# ╠═17de9582-76f2-11eb-03cf-43ad5b07d120
# ╟─f29e95dc-7d50-11eb-1ecb-51f578ef29f0
# ╠═01417a14-7d51-11eb-394e-a36a82d4a40a
# ╠═ade2915c-7709-11eb-2fce-45999f1e9ffa
# ╠═299e3eaa-7709-11eb-2b3e-13dcfbd9f450
# ╟─67034e64-76f2-11eb-2a25-c53adf9adde6
# ╠═7680d44c-76f2-11eb-1ffe-f92fdd5e42b7
# ╠═db3bbb6e-76f6-11eb-09c1-e7f0392c634d
# ╟─76610114-76f2-11eb-3e22-71a763599e47
# ╠═762adc54-76f2-11eb-259d-8f6cae34ef20
# ╠═75e24a66-76f2-11eb-2711-f53a6c04fb99
# ╠═75bfd17a-76f2-11eb-3285-31b3eaa96a62
# ╠═3b78e5bc-7707-11eb-1e32-2116b3156727
# ╠═b398fef4-7706-11eb-2b36-17f85f356a3c
# ╟─42d86fd8-7878-11eb-0e98-c73dd498e285
# ╠═861170cc-7877-11eb-00c7-0f50fe4ceeea
# ╟─d4844b50-770b-11eb-221c-0f281a8d8eec
# ╠═f40d002a-770b-11eb-3d91-8916193cf7a3
# ╟─ce050256-7aee-11eb-1474-b96dcad21315
# ╟─5fcd5f68-7715-11eb-3667-7d4960547cff
# ╠═187d5a38-7875-11eb-01c1-bbc44e1d15c0
# ╠═0f7839a6-787e-11eb-2887-f34311d11b56
# ╠═a6d8c4c2-771c-11eb-30bc-332df04ade43
# ╠═4dda42ea-774b-11eb-2136-b345e017b297
# ╠═9c2dd014-7751-11eb-1ecb-51f578ef29f0
# ╠═e265712c-7aee-11eb-28c8-9952ca7fbc46
# ╠═375e9688-7af2-11eb-08fe-efb6d6b91863
# ╠═ef617988-7d59-11eb-2f81-69e21d6c7ffa
# ╠═69aa0492-7d5c-11eb-2f81-69e21d6c7ffa
# ╠═27f66dec-7d4f-11eb-29a5-d79c586de1d8
# ╠═eced8604-7d44-11eb-246e-a77066a91eb1
# ╟─7e24bb38-770a-11eb-03ef-590069fda67c
# ╠═b3580318-7706-11eb-0b23-6b817d62e154
# ╠═909e2708-770b-11eb-0955-6f3c7c6d19ec
# ╟─b33d33f8-7706-11eb-1b83-130c4fb81eb2
# ╠═a572e232-770a-11eb-12ce-d5cf2e70d878
# ╠═e3809494-7af6-11eb-33b9-4d503e9aa1fc
# ╠═29716c86-7d53-11eb-1de9-7ba4e5be7112
# ╠═be5a7876-7d56-11eb-145f-45c61470f7cc
# ╠═46c01a6c-7d53-11eb-0e1b-cbc471d0a34e
# ╠═a556fdf6-770a-11eb-18c3-3fc37fe4fe32
# ╠═b30dea8a-7706-11eb-266b-eb9027195a0b
# ╠═fc20085e-7d53-11eb-2f81-69e21d6c7ffa
