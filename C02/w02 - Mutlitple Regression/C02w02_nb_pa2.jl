### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 1bcf25d8-71c2-11eb-05ca-6fe7c4e1b24f
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
end

# ╔═╡ 1f7ea604-71c2-11eb-028a-a166f6f4dc5d
begin
	using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	using Random
	using Test
	using Printf
end

# ╔═╡ f377f6d4-71c9-11eb-0249-f3d63e87062b
begin 
	using LinearAlgebra

	function feature_derivative(errors::Vector{T}, 
			feature::Vector{T}) where {T <: Real}
    	return 2 * dot(feature, errors) # ≡ 2 * feature' * errors
	end
end

# ╔═╡ 9dcffb1c-71c1-11eb-2402-39eaa1f1b5e1
md"""
## Week 2: Multiple Regression (Gradient Descent)
"""

# ╔═╡ c39f9348-71c1-11eb-1691-1d28c6b1d916
md"""

In this notebook we will cover estimating multiple regression weights via gradient descent, performing the following:

 - Add a constant column of 1's to a DataFrame to account for the intercept
 - Write a `predict_output()` function 
 - Write a function to compute the derivative of the regression weights with respect to a single feature
 - Write gradient descent function to compute the regression weights given an initial weight vector, step size and tolerance.
 - Use the gradient descent function to estimate regression weights for multiple features
"""

# ╔═╡ 3e5d907e-71c2-11eb-2eb3-4375fb21b884
md"""
### Load in house sales data
"""

# ╔═╡ 4c736078-71c2-11eb-225b-816745199225
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ 58e6255a-71c2-11eb-2094-9dbe7fb64151
first(sales, 3)

# ╔═╡ 64c97662-71c2-11eb-2894-2d544642b136
typeof(sales.bedrooms)

# ╔═╡ 80fbf0f8-71c2-11eb-265a-fb539d764180
typeof(sales.bathrooms)

# ╔═╡ 35b5d05a-71cc-11eb-15a9-852c5a534fab
md"""
### Convert to Julia Matrix/Vector
"""

# ╔═╡ a2928dc6-71c2-11eb-3724-29dc1448a0f3
md"""
Now we will write a function that will accept a DataFrame, a list of feature names (e.g. [:sqft_living, :bedrooms') and a target feature e.g. (:price) and will return two things:

 - A matrix whose columns are the desired features plus a constant column (this is how we create an 'intercept')
 - An array containing the values of the output

With this in mind, let's write the `get_data` function:

"""

# ╔═╡ e4da992e-71c2-11eb-36b8-93dd614051c1
function get_data(df, features, output)
	df[:, :constant] .= 1.0 # df.constant = fill(1.0, size(df, 1))
	features = [:constant, features...]
	X_matrix = convert(Matrix, select(df, features)) # to get a matrix 
	y = df[!, output]                                # => to get a vector
	return (X_matrix, y)
end

# ╔═╡ d62e0368-71c2-11eb-16f7-5fda56951d21
begin
	(ex_features, ex_output) = get_data(sales, [:sqft_living], :price) 
	with_terminal() do
		println(ex_features[1:5, :], " / ", typeof(ex_features))
		println(ex_output[1], " / ", typeof(ex_output))
	end
end

# ╔═╡ 498b3458-71cc-11eb-3b55-8381d58d04bb
md"""
### Predicting output given regression weights
"""

# ╔═╡ daf8aefc-71c7-11eb-04fa-e71d7b2cde21
md"""
The predictions from all the observations are just the dot product between the features matrix (on the left) and the weights vector (on the right).

With this in mind write the following `predict_output` function to compute the predictions given the feature matrix (X) and the weights:
"""

# ╔═╡ 02f0c99e-71c5-11eb-1fe1-b3f820998d54
function predict_output(X::Matrix{T}, weights::Vector{T}) where {T <: Real}
    # assume feature_matrix is a matrix containing the features as columns
	# and weights is a corresponding array
    X * weights
end

# ╔═╡ 9cd636f2-71c5-11eb-2957-afbaf437aaa5
begin
	my_weights = Float64[1., 1.]
	test_predictions = predict_output(ex_features, my_weights)
	
	@test test_predictions[1] == 1431.0  # should be 1431.0
	@test test_predictions[2] == 2951.0  # should be 2951.0
end

# ╔═╡ 21635b6e-71c8-11eb-2272-edec98c58339
md"""
### Computing the derivative


We are now going to compute the derivative of the regression cost function. Recall that the cost function is the sum over the data points of the squared difference between an observed output and a predicted output.

Since the derivative of a sum is the sum of the derivatives we can compute the derivative for a single data point and then sum over data points. We can write the squared difference between the observed output and predicted output for a single point as follows:

$$(w[0]\times[CONST] + w[1]\times[feature_1] + ... + w[i]\times[feature_i] + ... +  w[k]\times[feature_k] - output)^2$$

Where we have $k$ features and a constant. So the derivative with respect to weight $w[i]$ by the chain rule is:

$$2\times(w[0]\times[CONST] + w[1]\times[feature_1] + ... + w[i]\times[feature_i] + ... +  w[k]\times[feature_k] - output)\times[feature_i]$$

The term inside the parenthesis is just the error (difference between prediction and output). So we can re-write this as:

$$2\times error\times[feature_i]$$

That is, the derivative for the weight for feature $i$ is:
  - the sum (over data points) of 2 × the product of the error and the feature itself.   
    In the case of the constant then this is just twice the sum of the errors!

Recall that twice the sum of the product of two vectors is just:
  - twice the dot product of the two vectors.  
    Therefore the derivative for the weight for $feature_i$ is just two times the dot product between the values of $feature_i$ and the current errors. 

With this in mind let's write the following derivative function which computes the derivative of the weight given the value of the feature (over all data points) and the errors (over all data points).


`doc ref. https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/`
"""

# ╔═╡ 1301a4d2-71ca-11eb-339c-d1f747a3ffe7
begin
	(nex_features, nex_output) = get_data(sales, [:sqft_living], :price) 
	ya_weights = Float64[0., 0.] # this makes all the predictions 0
	test_preds = predict_output(nex_features, ya_weights) 

	# just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
	errors = test_preds .- nex_output # prediction errors in this case is just the -example_output
	ya_feature = nex_features[:, 1]   # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
	der = feature_derivative(errors, ya_feature)
	alt_der = -sum(nex_output) * 2

	@test der == alt_der  # should be the same as derivative
end

# ╔═╡ 49ce0c84-71cb-11eb-3947-6141a9498ffb
with_terminal() do
	@printf("derivative: %2.5e / alt_derivative: %2.5e\n", der, alt_der)
end

# ╔═╡ 239ff6ca-71cc-11eb-341a-df1453d60a15
md"""
### Gradient Descent [GD]


Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Recall that the gradient is the direction of increase and therefore the negative gradient is the direction of decrease and we're trying to minimize a cost function.

The amount by which we move in the negative gradient direction is called the *step size* denoted by η. We stop when we are 'sufficiently close' to the optimum. We define this by requiring that the magnitude (length) of the gradient vector to be smaller than a fixed *tolerance* denoted by ϵ.

With this in mind, wriet the following gradient descent function below using the derivative function above. For each step in the gradient descent we update the weight for each feature befofe computing our stopping criteria
"""

# ╔═╡ 958f81a6-71cc-11eb-13e8-61dddeb014a7
function regression_gradient_descent(f_matrix::Matrix{T}, 
		output::Vector{T}, init_weights::Vector{T}, η::T, ϵ::T) where {T <: Real}
    weights = init_weights # make sure it's a vector
	
    while true
		preds = predict_output(f_matrix, weights) # compute the predictions 
        errors = preds .- output                  # compute the errors 
        ∇_sum_squares = 0.0                       # init the gradient sum of squares
        
		## Update the weights
        for ix ∈ 1:length(weights) # loop over each weight
            ## Recall that feature_matrix[:, ix] is the feature column 
		    ## associated with weights[i]
            deriv_ix = feature_derivative(errors, f_matrix[:, ix]) 
			## add squared value of derivative to ∇_sum_squares (convergence)
            ∇_sum_squares += deriv_ix * deriv_ix 
			
			## subtract the step size times the derivative from the CURRENT weight
            weights[ix] -= η * deriv_ix
		end
        ##  compute the square-root of the ∇_sum_squares to get the ∇ magnitude:
        ∇_magnitude = √(∇_sum_squares)
        ∇_magnitude < ϵ && break
	end    
    weights
end

# ╔═╡ 4454239e-71ce-11eb-295e-1755bc0549e6
md"""
### Running the Gradient Descent as Simple Regression
"""

# ╔═╡ 0623d920-71ce-11eb-0936-870685f2e788
begin
function train_test_split(df; split=0.8, seed=42, shuffled=true) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffled ? shuffle(1:nr) : 1:nr
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	(df_train, df_test)
end

sales_train, sales_test = train_test_split(sales);
end

# ╔═╡ 6a1ee460-71ce-11eb-1b49-4de942aefe9d
begin
	# let's test out the gradient descent
	s_features = [:sqft_living]
	s_out = :price
	(simple_f_matrix, out) = get_data(sales_train, s_features, s_out)

	init_weights = [-47000., 1.]
	η = 7e-12
	ϵ = 2.5e7
end

# ╔═╡ 9883c1ee-71cf-11eb-3b48-a14df232588f
gd_weights = regression_gradient_descent(simple_f_matrix, Vector{Float64}(out), init_weights, η, ϵ)

# ╔═╡ 06cf79e0-71d0-11eb-3eab-ad040213a64b
md"""
Use your newly estimated weights and your `predict_output()` function to compute the predictions on all the test data.

you will need to create a julia array of the test feature_matrix and test output first, as follows:
"""

# ╔═╡ 26199ccc-71d0-11eb-0b4e-5f2f8ebc7b76
(test_s_feature_matrix, test_out) = get_data(sales_test, s_features, s_out);

# ╔═╡ 4be11b60-71d0-11eb-1550-6d2854691b05
md"""
Now compute your predictions using `test_s_feature_matrix` and the `gd_weights` from above.
"""

# ╔═╡ 65c26930-71d0-11eb-248a-3751d927b2f6
ya_preds = predict_output(test_s_feature_matrix, gd_weights)

# ╔═╡ 749979f6-71d0-11eb-0b3a-459257d95ca9
md"""
**Quiz Question: What is the predicted price for the 1st house in the TEST data set for model 1 (round to nearest dollar)?**

"""

# ╔═╡ 898fa5b2-71d0-11eb-18c3-ad6d0dbf9bf8
begin
	pred_price_first_house = ya_preds[1]
	
	with_terminal() do
		@printf("price: %6.2f / price rounded to nearest dollar: %6.0f\n", pred_price_first_house, round(pred_price_first_house))
	end
end

# ╔═╡ 5f0bdea2-71d3-11eb-150c-8722eee3f23a
md"""
Now that you have the predictions on test data, compute the RSS on the test data set. Save this value for comparison later. Recall that RSS is the sum of the squared errors (difference between prediction and output).
"""

# ╔═╡ 6d6eb4b0-71d3-11eb-24f0-f9f796040473
begin
	rss_test = sum((ya_preds - test_out).^2)

	with_terminal() do
		@printf("rss on test set: %15.2f / in scientific notation: %2.4e\n", rss_test, rss_test)
	end
end

# ╔═╡ e13d4ca6-71d0-11eb-1ad7-119b4bf3c830
md"""
### Running a multiple regression

Now we will use more than one actual feature. Use the following code to produce the weights for a second model with the following parameters:

"""

# ╔═╡ ec505642-71d0-11eb-2619-0f410fee1f94
begin
	mr_model_features = [:sqft_living, :sqft_living15] 
	mr_target = :price
	(mr_feature_matrix, mr_output) = get_data(sales_train, mr_model_features,
			mr_target)
	mr_init_weights = [-100000., 1., 1.]
	ηᵣ = 4e-12
	ϵᵣ = 1e9
end

# ╔═╡ 59a6c7bc-71d1-11eb-2836-71ec957bd538
md"""
Use the above parameters to estimate the model weights. Record these values for your quiz.
"""

# ╔═╡ 6a907c30-71d1-11eb-3adb-1355e084f741
weights_mr = regression_gradient_descent(mr_feature_matrix, 
	Vector{Float64}(mr_output), mr_init_weights, ηᵣ, ϵᵣ)

# ╔═╡ c47ea9ec-71d1-11eb-0491-9ba85adf1e0a
md"""
Use your newly estimated weights and the predict_output function to compute the predictions on the test data.

*Don't forget to create a Julia array for these features from the test set first!*
"""

# ╔═╡ da6f7c6a-71d1-11eb-381f-5f75054e14e7
begin
	(test_mr_features_matrix, test_mr_output) = get_data(sales_test, 
		mr_model_features, mr_target)

	preds_mr = predict_output(test_mr_features_matrix, weights_mr)
end

# ╔═╡ 05450824-71d2-11eb-0a05-2b895313421f
md"""
**Quiz Question: What is the predicted price for the 1st house in the TEST data set for model 2 (round to nearest dollar)?**

"""

# ╔═╡ 1706d680-71d2-11eb-3cd6-4d156d964fd7
begin
	pred_mr_price_first_house = preds_mr[1] 


	with_terminal() do
		@printf("price: %6.2f / price rounded to nearest dollar: %6.0f\n", pred_mr_price_first_house, round(pred_mr_price_first_house))
	end
end

# ╔═╡ 44c15f32-71d2-11eb-2232-5fd5a07b2bfc
md"""
What is the actual price for the 1st house in the test data set?
"""

# ╔═╡ 4eeb7218-71d2-11eb-1fa7-21fec4e8c846
sales_test[!, :price][1]

# ╔═╡ 84462372-71d2-11eb-1627-1fc0ccb0d650
md"""
**Quiz Question: Which estimate was closer to the true price for the 1st house on the *test* data set, model 1 or model 2?**

  - [x] Model 1
  - [ ] Model 2

"""

# ╔═╡ 8ef08466-71d2-11eb-0cbd-17d088b840e0
md"""
Now use your predictions and the output to compute the RSS for model 2 on *test* data.
"""

# ╔═╡ 9db31f5e-71d2-11eb-1527-29c0a844ce47
begin
	rss_test_mr = sum((preds_mr - test_mr_output).^2)

	with_terminal() do
		@printf("rss on test set: %15.2f / in scientific notation: %2.4e\n", rss_test_mr, rss_test_mr)
	end
end

# ╔═╡ 0d254b32-71d3-11eb-2c92-b1a2d1125c5b
md"""
**Quiz Question: Which model (1 or 2) has lowest RSS on all of the test data?**

  - [ ] model 1
  - [x] model 2
"""

# ╔═╡ Cell order:
# ╟─9dcffb1c-71c1-11eb-2402-39eaa1f1b5e1
# ╟─c39f9348-71c1-11eb-1691-1d28c6b1d916
# ╠═1bcf25d8-71c2-11eb-05ca-6fe7c4e1b24f
# ╠═1f7ea604-71c2-11eb-028a-a166f6f4dc5d
# ╟─3e5d907e-71c2-11eb-2eb3-4375fb21b884
# ╠═4c736078-71c2-11eb-225b-816745199225
# ╠═58e6255a-71c2-11eb-2094-9dbe7fb64151
# ╠═64c97662-71c2-11eb-2894-2d544642b136
# ╠═80fbf0f8-71c2-11eb-265a-fb539d764180
# ╟─35b5d05a-71cc-11eb-15a9-852c5a534fab
# ╟─a2928dc6-71c2-11eb-3724-29dc1448a0f3
# ╠═e4da992e-71c2-11eb-36b8-93dd614051c1
# ╠═d62e0368-71c2-11eb-16f7-5fda56951d21
# ╟─498b3458-71cc-11eb-3b55-8381d58d04bb
# ╟─daf8aefc-71c7-11eb-04fa-e71d7b2cde21
# ╠═02f0c99e-71c5-11eb-1fe1-b3f820998d54
# ╠═9cd636f2-71c5-11eb-2957-afbaf437aaa5
# ╟─21635b6e-71c8-11eb-2272-edec98c58339
# ╠═f377f6d4-71c9-11eb-0249-f3d63e87062b
# ╠═1301a4d2-71ca-11eb-339c-d1f747a3ffe7
# ╠═49ce0c84-71cb-11eb-3947-6141a9498ffb
# ╟─239ff6ca-71cc-11eb-341a-df1453d60a15
# ╠═958f81a6-71cc-11eb-13e8-61dddeb014a7
# ╟─4454239e-71ce-11eb-295e-1755bc0549e6
# ╠═0623d920-71ce-11eb-0936-870685f2e788
# ╠═6a1ee460-71ce-11eb-1b49-4de942aefe9d
# ╠═9883c1ee-71cf-11eb-3b48-a14df232588f
# ╟─06cf79e0-71d0-11eb-3eab-ad040213a64b
# ╠═26199ccc-71d0-11eb-0b4e-5f2f8ebc7b76
# ╟─4be11b60-71d0-11eb-1550-6d2854691b05
# ╠═65c26930-71d0-11eb-248a-3751d927b2f6
# ╟─749979f6-71d0-11eb-0b3a-459257d95ca9
# ╠═898fa5b2-71d0-11eb-18c3-ad6d0dbf9bf8
# ╠═5f0bdea2-71d3-11eb-150c-8722eee3f23a
# ╠═6d6eb4b0-71d3-11eb-24f0-f9f796040473
# ╟─e13d4ca6-71d0-11eb-1ad7-119b4bf3c830
# ╠═ec505642-71d0-11eb-2619-0f410fee1f94
# ╟─59a6c7bc-71d1-11eb-2836-71ec957bd538
# ╠═6a907c30-71d1-11eb-3adb-1355e084f741
# ╟─c47ea9ec-71d1-11eb-0491-9ba85adf1e0a
# ╠═da6f7c6a-71d1-11eb-381f-5f75054e14e7
# ╟─05450824-71d2-11eb-0a05-2b895313421f
# ╠═1706d680-71d2-11eb-3cd6-4d156d964fd7
# ╟─44c15f32-71d2-11eb-2232-5fd5a07b2bfc
# ╠═4eeb7218-71d2-11eb-1fa7-21fec4e8c846
# ╟─84462372-71d2-11eb-1627-1fc0ccb0d650
# ╟─8ef08466-71d2-11eb-0cbd-17d088b840e0
# ╠═9db31f5e-71d2-11eb-1527-29c0a844ce47
# ╟─0d254b32-71d3-11eb-2c92-b1a2d1125c5b
