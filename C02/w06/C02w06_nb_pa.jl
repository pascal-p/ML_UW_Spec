### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ cbcca572-7614-11eb-2fe5-699acc8e2746
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
	
	# using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	# using Random
	using Test
	using Printf
	using Plots
end


# ╔═╡ f1eefcb4-7614-11eb-0173-d5112c9036d7
include("./utils.jl")

# ╔═╡ 84a95dde-7614-11eb-20da-8191d2c5103e
md"""
## Predicting house prices using k-nearest neighbors regression

In this notebook, we will implement k-nearest neighbors regression. We will:
  - Find the k-nearest neighbors of a given query input
  - Predict the output for the query input using the k-nearest neighbors
  - Choose the best value of k using a validation set
"""

# ╔═╡ db51098e-7614-11eb-2dcd-57f4526747f6
md"""
### Load in house sales data (train, valid and test sets)
"""

# ╔═╡ f20a9c80-7614-11eb-1400-e55b390c3437
begin
  	train = CSV.File("../../ML_UW_Spec/C02/data/kc_house_data_small_train.csv"; 
		header=true) |> DataFrame;
  	valid = CSV.File("../../ML_UW_Spec/C02/data/kc_house_data_small_validation.csv"; 
		header=true) |> DataFrame;
	test = CSV.File("../../ML_UW_Spec/C02/data/kc_house_data_small_test.csv"; 
		header=true) |> DataFrame;

	size(train), size(valid), size(test)
end

# ╔═╡ e0ad9bee-7615-11eb-0226-617fef643e14
md"""
We need `get_data()` and `normalize_features()` from previous week. Let's import them from our utils module.
"""

# ╔═╡ f1d25e60-7614-11eb-2db3-69a5c37b2856
md"""
### Extract features and normalize

Using all of the numerical inputs listed in feature_list, transform the training, test, and validation DataFrames into Julia arrays:
"""

# ╔═╡ f1a1aa7c-7614-11eb-24b8-3183535eb4c1
md"""
In computing distances, it is crucial to normalize features. Otherwise, for example, the sqft_living feature (typically on the order of thousands) would exert a much larger influence on distance than the bedrooms feature (typically on the order of ones). We divide each column of the training feature matrix by its 2-norm, so that the transformed column has unit norm.

*IMPORTANT*: Make sure to store the norms of the features in the training set. The features in the test and validation sets must be divided by these same norms, so that the training, test, and validation sets are normalized consistently.

"""

# ╔═╡ f1b90668-7614-11eb-1313-0b2a3b91b243
begin
	feature_list = [:bedrooms, :bathrooms, 
                :sqft_living, :sqft_lot,  
                :floors, :waterfront, 
                :view, :condition,  
                :grade, :sqft_above,  
                :sqft_basement, :yr_built,  
                :yr_renovated, :lat, :long,  
                :sqft_living15, :sqft_lot15]

	features_train, output_train = get_data(train, feature_list, :price)
	features_test, output_test = get_data(test, feature_list, :price)
	features_valid, output_valid = get_data(valid, feature_list, :price)
	
	## Normalize
	features_train, norms = normalize_features(features_train) # norm. train set features (columns)
	features_test = features_test ./ norms    # normalize test set ...
	features_valid = features_valid ./ norms  # ... and valid set with training set norms
	
	size(features_train), size(features_test), size(features_valid)
end

# ╔═╡ 0249baea-761a-11eb-09fa-efa1de647bb8
md"""
### Compute a single distance

To start, let's just explore computing the "distance" between two given houses. We will take our query house to be the first house of the test set and look at the distance between this house and the 10th house of the training set.

To see the features associated with the query house, print the first row (index 1) of the test feature matrix. You should get an 18-dimensional vector whose components (in absolute value) are between 0 and 1.

"""

# ╔═╡ 022e64ac-761a-11eb-1daf-d9ad5926c387
begin
	qry_house = features_test[1, :] 
	
	@test size(qry_house)[1] == 18
	@test all(x -> 0. ≤ abs(x) ≤ 1., qry_house)
end

# ╔═╡ 020058f8-761a-11eb-0b03-1be1e9780d1c
md"""
**Quiz Question**

What is the Euclidean distance between the query house and the 10th house of the training set?

"""

# ╔═╡ f15192e4-7614-11eb-1fca-3b6fc4634159
begin
	function calc_euclidean_distance(row::Vector, row_ref::Vector)
  		√(sum((row - row_ref) .* (row - row_ref)))
	end
	
	function calc_euclidean_distance(rows::Matrix, row_ref::Vector)
		sqrt.(sum(((rows .- row_ref').^2), dims=2))
	end
end

# ╔═╡ f137c3e6-7614-11eb-190e-4d331f41b02e
begin
	d10th = calc_euclidean_distance(features_train[10, :], qry_house)
	
	with_terminal() do
		@printf("distance %1.8f / rounded: %1.3f\n", d10th, d10th)
	end
end

# ╔═╡ 25250e4c-761b-11eb-2c45-f54d7410184a
md"""
### Compute multiple distances

Of course, to do nearest neighbor regression, we need to compute the distance between our query house and all houses in the training set.

To visualize this nearest-neighbor search, let's first compute the distance from our query house (features_test[1]) to the first 10 houses of the training set (features_train[1:10]) and then search for the nearest neighbor within this small set of houses. Through restricting ourselves to a small set of houses to begin with, we can visually scan the list of 10 distances to verify that our code for finding the nearest neighbor is working.

Compute the Euclidean distance from the query house to each of the first 10 houses in the training set.
"""

# ╔═╡ 67931b0c-761b-11eb-3f3c-cfa948d07ebf
dist_10 = calc_euclidean_distance(features_train[1:10, :], qry_house)

# ╔═╡ e504f550-761f-11eb-1471-0b3ae40edef8
sort(collect(enumerate(dist_10[:,])), 
	by=((_ix, x)=t) -> x, rev=false)[1:2]

# ╔═╡ 67441dd6-761b-11eb-1008-9f7fe69785b8
md"""
**Quiz Question**

Among the first 10 training houses, which house is the closest to the query house?
"""

# ╔═╡ 6699cfcc-761b-11eb-0c6a-89637d586437
## Resp.
begin
	ix_10 = argmin(dist_10)
	d_10 = dist_10[ix_10]
	(index=ix_10[1], distance=d_10)
end

# ╔═╡ 667cb82c-761b-11eb-1925-153041dc9a6a
md"""
### Perform k-nearest neighbor regression

For k-nearest neighbors, we need to find a set of k houses in the training set closest to a given query house. We then make predictions based on these k nearest neighbors.
Fetch k-nearest neighbors

Using the functions above, implement a function that takes in
  - the value of k;
  - the feature matrix for the training houses; and
  - the feature vector of the query house

and returns the indices of the k closest training houses. For instance, with 2-nearest neighbor, a return value of [5, 10] would indicate that the 5th and 10th training houses are closest to the query house.
"""

# ╔═╡ 6661ff46-761b-11eb-2817-0bf3e5c48275
function k_nearest(k::Int, feature_matrix::Matrix, feature_vect::Vector)
	@assert k ≥ 1
  	dist = calc_euclidean_distance(feature_matrix, feature_vect)
	sort(collect(enumerate(dist[:])), 
		by=((_ix, x)=t) -> x, rev=false)[1:k] |>
		a -> map(t -> t[1], a)
end

# ╔═╡ 6645eb26-761b-11eb-2de2-212f9a229bef
md"""
**Quiz Question**

Take the query house to be third house of the test set (features_test[3]). What are the indices of the 4 training houses closest to the query house?
"""

# ╔═╡ 6628a162-761b-11eb-386d-5d6c4b9d8646
## for quiz, index answers are expected to be 0-based
## Julia however is 1-based index, thus removed 1 to this answers
k_nearest(4, features_train, features_test[3, :])

# ╔═╡ 660da98c-761b-11eb-34af-b3b35c66288c
md"""
### Make a single prediction by averaging k nearest neighbor outputs

Now that we know how to find the k-nearest neighbors, write a function that predicts the value of a given query house. For simplicity, take the average of the prices of the k nearest neighbors in the training set. The function should have the following parameters:

  - the value of k;
  - the feature matrix for the training houses;
  - the output values (prices) of the training houses; and
  - the feature vector of the query house, whose price we are predicting.

The function should return a predicted value of the query house.

"""

# ╔═╡ f7a44f48-7622-11eb-1d99-e33a5cbe3f29
function predict(k::Int, feature_matrix::Matrix, feature_vect::Vector, 
		output::Vector)
  ixes = k_nearest(k, feature_matrix, feature_vect)
  sum(output[ixes]) / k
end

# ╔═╡ f7809ec2-7622-11eb-2d16-615ec1f7fb49
md"""
### Quiz Question

Again taking the query house to be third house of the test set (features_test[3]), predict the value of the query house using k-nearest neighbors with k=4 and the simple averaging method described and implemented above.
"""

# ╔═╡ f7637df6-7622-11eb-076f-0f6ab652bfd2
predict(4, features_train, features_test[3, :], output_train)
# cmp
# 413987.5 (k-nearest) vs 249000 (1-nearest)

# ╔═╡ f74b6982-7622-11eb-37d2-21ed0ec6551d
md"""
### Make multiple predictions

Write a function to predict the value of each and every house in a query set. (The query set can be any subset of the dataset, be it the test set or validation set.) The idea is to have a loop where we take each house in the query set as the query house and make a prediction for that specific house. The new function should take the following parameters:

  - the value of k;
  - the feature matrix for the training houses;
  - the output values (prices) of the training houses; and
  - the feature matrix for the query set.

The function should return a set of predicted values, one for each house in the query set.

"""

# ╔═╡ f728dac0-7622-11eb-094f-6782145366d8
function predict_all(k::Int, feature_matrix::Matrix, feature_matrix_qs::Matrix,
		output::Vector)
	n = size(feature_matrix_qs)[1]
	vresp = zeros(Float64, n)
	for ix ∈ 1:n
		vresp[ix] = predict(k, feature_matrix, feature_matrix_qs[ix, :], output)
	end
	vresp
end

# ╔═╡ 5b3578e0-7626-11eb-1c95-27a497721fd1
size(features_test), typeof(features_test)

# ╔═╡ c1c949fc-7625-11eb-212e-7316c164b585
size(features_test[1:10, :]), typeof(features_test[1:10, :])

# ╔═╡ a307a67c-7624-11eb-0021-09f621585464
size(features_test[1:10, :])[1]

# ╔═╡ c51c48a4-7628-11eb-07cd-97f29b9fb5fa
output_train

# ╔═╡ 65f46ecc-761b-11eb-1617-0fe428d5a1bd
md"""
**Quiz Question**

Make predictions for the first 10 houses in the test set using k-nearest neighbors with k=10.

  1. What is the index of the house in this query set that has the lowest predicted value?
  2. What is the predicted value of this house?

"""

# ╔═╡ d4f46ee6-7625-11eb-3b88-3d68b2e1c58b
features_test[2, :], size(features_test[2, :])

# ╔═╡ 65dba81a-761b-11eb-3bb8-47bf75fafa40
begin
	sol = predict_all(10, features_train, features_test[1:10, :], output_train)
	m_ix = argmin(sol)
	with_terminal() do
		println((solution=sol, index=m_ix[1], predicted_price=sol[m_ix]))
	end
end

# ╔═╡ 65bfee54-761b-11eb-16e8-4b4ca6decc9d
md"""
### Choosing the best value of k using a validation set

There remains a question of choosing the value of k to use in making predictions. Here, we use a validation set to choose this value. Write a loop that does the following:

```
  For k ∈ [1, 2, ..., 15]
    Makes predictions for each house in the validation set using the k-nearest neighbors from the training set.
    Computes the RSS for these predictions on the validation set
    Stores the RSS computed above in rss_all
  End
  Report which k produced the lowest RSS on validation set.
```

"""

# ╔═╡ 9cb7ace4-762a-11eb-3e80-f16cbf712a62
function rss_fn(preds, output)
	sum((preds .- output) .^ 2)
end

# ╔═╡ 726f0674-7629-11eb-2c27-7d634b6d1d4b
function find_best_k(features_matrix::Matrix, features_matrix_qs::Matrix, 
		output::Vector, output_val::Vector; range=1:15)
 	rss_all = []
  	best_k, best_rss = nothing, nothing
  	for k ∈ range
    	preds = predict_all(k, features_matrix, features_matrix_qs, output)
    	rss = rss_fn(preds, output_val)
    	push!(rss_all, rss)
    	if isnothing(best_rss) || rss < best_rss
			best_rss, best_k = rss, k
		end
	end
  	(best_k, best_rss, rss_all)
end

# ╔═╡ 65a2c806-761b-11eb-2f1c-3773d2584857
(best_k, best_rss, rss_all) = find_best_k(features_train, features_valid, output_train, output_valid)

# ╔═╡ 6577f8ec-761b-11eb-01f5-67cb002a7070
md"""
To visualize the performance as a function of k, plot the RSS on the *validation* set for each considered k value.
"""

# ╔═╡ 377da4be-762a-11eb-31be-c1041f7c306b
plot(collect(1:15), rss_all, legend=false, color=[:lightblue], marker=".")

# ╔═╡ 754a565c-762a-11eb-0b81-9d30ddf27406
md"""
**Quiz Question**

What is the RSS on the test data using the value of k found above? To be clear, sum over all houses in the test set.

"""

# ╔═╡ 9210589a-762a-11eb-3439-f3130e570162
begin
	test_preds = predict_all(best_k, features_train, features_test, output_train)
	test_rss = rss_fn(test_preds, output_test)
	
	with_terminal() do
		@printf("test rss: %1.7e\n", test_rss)
	end
end

# ╔═╡ Cell order:
# ╟─84a95dde-7614-11eb-20da-8191d2c5103e
# ╠═cbcca572-7614-11eb-2fe5-699acc8e2746
# ╟─db51098e-7614-11eb-2dcd-57f4526747f6
# ╠═f20a9c80-7614-11eb-1400-e55b390c3437
# ╟─e0ad9bee-7615-11eb-0226-617fef643e14
# ╠═f1eefcb4-7614-11eb-0173-d5112c9036d7
# ╟─f1d25e60-7614-11eb-2db3-69a5c37b2856
# ╟─f1a1aa7c-7614-11eb-24b8-3183535eb4c1
# ╠═f1b90668-7614-11eb-1313-0b2a3b91b243
# ╟─0249baea-761a-11eb-09fa-efa1de647bb8
# ╠═022e64ac-761a-11eb-1daf-d9ad5926c387
# ╟─020058f8-761a-11eb-0b03-1be1e9780d1c
# ╠═f15192e4-7614-11eb-1fca-3b6fc4634159
# ╠═f137c3e6-7614-11eb-190e-4d331f41b02e
# ╟─25250e4c-761b-11eb-2c45-f54d7410184a
# ╠═67931b0c-761b-11eb-3f3c-cfa948d07ebf
# ╠═e504f550-761f-11eb-1471-0b3ae40edef8
# ╟─67441dd6-761b-11eb-1008-9f7fe69785b8
# ╠═6699cfcc-761b-11eb-0c6a-89637d586437
# ╟─667cb82c-761b-11eb-1925-153041dc9a6a
# ╠═6661ff46-761b-11eb-2817-0bf3e5c48275
# ╟─6645eb26-761b-11eb-2de2-212f9a229bef
# ╠═6628a162-761b-11eb-386d-5d6c4b9d8646
# ╟─660da98c-761b-11eb-34af-b3b35c66288c
# ╠═f7a44f48-7622-11eb-1d99-e33a5cbe3f29
# ╟─f7809ec2-7622-11eb-2d16-615ec1f7fb49
# ╠═f7637df6-7622-11eb-076f-0f6ab652bfd2
# ╟─f74b6982-7622-11eb-37d2-21ed0ec6551d
# ╠═f728dac0-7622-11eb-094f-6782145366d8
# ╠═5b3578e0-7626-11eb-1c95-27a497721fd1
# ╠═c1c949fc-7625-11eb-212e-7316c164b585
# ╠═a307a67c-7624-11eb-0021-09f621585464
# ╠═c51c48a4-7628-11eb-07cd-97f29b9fb5fa
# ╟─65f46ecc-761b-11eb-1617-0fe428d5a1bd
# ╠═d4f46ee6-7625-11eb-3b88-3d68b2e1c58b
# ╠═65dba81a-761b-11eb-3bb8-47bf75fafa40
# ╟─65bfee54-761b-11eb-16e8-4b4ca6decc9d
# ╠═9cb7ace4-762a-11eb-3e80-f16cbf712a62
# ╠═726f0674-7629-11eb-2c27-7d634b6d1d4b
# ╠═65a2c806-761b-11eb-2f1c-3773d2584857
# ╟─6577f8ec-761b-11eb-01f5-67cb002a7070
# ╠═377da4be-762a-11eb-31be-c1041f7c306b
# ╟─754a565c-762a-11eb-0b81-9d30ddf27406
# ╠═9210589a-762a-11eb-3439-f3130e570162
