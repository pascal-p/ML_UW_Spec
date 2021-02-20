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
	using Plots
end

# ╔═╡ 96e7042c-73be-11eb-3ac5-9f9998b8c8ff
using LinearAlgebra

# ╔═╡ 9dcffb1c-71c1-11eb-2402-39eaa1f1b5e1
md"""
## Week 4: Ridge Regression (Gradient Descent)
"""

# ╔═╡ c39f9348-71c1-11eb-1691-1d28c6b1d916
md"""
In this notebook we will cover estimating multiple regression weights via gradient descent, performing the following:

  - Write a function to compute the derivative of the regression weights with respect to a single feature
  -  Write gradient descent function to compute the regression weights given an initial weight vector, step size, tolerance, and L2 penalty
"""

# ╔═╡ 3e5d907e-71c2-11eb-2eb3-4375fb21b884
md"""
### Load in house sales data
"""

# ╔═╡ 4c736078-71c2-11eb-225b-816745199225
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ 0623d920-71ce-11eb-0936-870685f2e788
function train_test_split(df; split=0.8, seed=42, shuffled=true) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffled ? shuffle(1:nr) : collect(1:nr)
	
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	
	(df_train, df_test)
end

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
	df[:, :constant] .= 1.0
	features = [:constant, features...]
	X_matrix = convert(Matrix, select(df, features)) # to get a matrix 
	y = df[!, output]                                # => to get a vector
	(X_matrix, y)
end

# ╔═╡ 02f0c99e-71c5-11eb-1fe1-b3f820998d54
function predict_output(X::Matrix{T}, weights::Vector{T}) where {T <: Real}
    # assume feature_matrix is a matrix containing the features as columns
	# and weights is a corresponding array
    X * weights
end

# ╔═╡ 21635b6e-71c8-11eb-2272-edec98c58339
md"""
### Computing the derivative

We are now going to move to computing the derivative of the regression cost function. Recall that the cost function is the sum over the data points of the squared difference between an observed output and a predicted output, plus the L2 penalty term.
```
Cost(w) = SUM[ (prediction - output)^2 ] + l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).
```

Since the derivative of a sum is the sum of the derivatives, we can take the derivative of the first part (the RSS) as we did in the notebook for the unregularized case in Week 2 and add the derivative of the regularization part. <br />
As we saw, the derivative of the RSS with respect to `w[i]` can be written as: 
```
2*SUM[ error*[feature_i] ]
```
The derivative of the regularization term with respect to `w[i]` is:
```
2*l2_penalty*w[i].
```
Summing both, we get
```
2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].
```
That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the feature itself, plus `2*l2_penalty*w[i]`. 

**We will not regularize the constant.**  Thus, in the case of the constant, the derivative is just twice the sum of the errors (without the `2*l2_penalty*w[0]` term).

Recall that twice the sum of the product of two vectors is just twice the dot product of the two vectors. Therefore the derivative for the weight for feature_i is just two times the dot product between the values of feature_i and the current errors, plus `2*l2_penalty*w[i]`.

With this in mind complete the following derivative function which computes the derivative of the weight given the value of the feature (over all data points) and the errors (over all data points).  
To decide when to we are dealing with the constant (so we don't regularize it) we added the extra parameter to the call `feature_is_constant` which you should set to `True` when computing the derivative of the constant and `False` otherwise.

"""

# ╔═╡ f377f6d4-71c9-11eb-0249-f3d63e87062b
function feature_derivative_ridge(errors::Vector{T}, feature::Vector{T}, 
		weights, l2_penalty; is_constant=false) where {T <: Real}
	deriv_err = 2 * dot(feature, errors) # ≡ 2 * feature' * errors
	deriv_l2p = l2_penalty * weights 
    return is_constant ? deriv_err : deriv_err + deriv_l2p
end

# ╔═╡ 59e3f22a-73bd-11eb-1c0d-150d7d462676
begin
	(ex_features, ex_output) = get_data(sales, [:sqft_living], :price)
	typeof(ex_features), typeof(ex_output)
	ex_weights = Float64[1., 10.]
	test_preds = predict_output(ex_features, ex_weights) 
	errors = test_preds - ex_output # prediction errors

	@test feature_derivative_ridge(errors, ex_features[:, 2], ex_weights[2], 1;
		is_constant=false) ≈ sum(errors .* ex_features[:, 2]) * 2. + 20.

	
	@test feature_derivative_ridge(errors, ex_features[:, 1], ex_weights[1], 1;
		is_constant=true) ≈ sum(errors) * 2.
end

# ╔═╡ 239ff6ca-71cc-11eb-341a-df1453d60a15
md"""
### Gradient Descent [GD]

Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Recall that the gradient is the direction of increase and therefore the negative gradient is the direction of decrease and we're trying to minimize a cost function.

The amount by which we move in the negative gradient direction is called the *step size* denoted by η. We stop when we are 'sufficiently close' to the optimum. Unlike in Week 2, this time we will set a maximum number of iterations and take gradient steps until we reach this maximum number. If no maximum number is supplied, the maximum should be set 100 by default. (Use default parameter values in Julia.)

With this in mind, write the following gradient descent function below using your derivative function above. For each step in the gradient descent, we update the weight for each feature before computing our stopping criteria.
"""

# ╔═╡ 958f81a6-71cc-11eb-13e8-61dddeb014a7
function regression_gradient_descent(f_matrix::Matrix{T}, 
		output::Vector{T}, init_weights::Vector{T}, η::T, l2_penalty; 
		max_iter=100) where {T <: Real}
    weights = copy(init_weights)
	iter = 0
	
    while iter < max_iter
		iter += 1
		preds = predict_output(f_matrix, weights) # compute the predictions 
        errors = preds .- output                  # compute the errors 
        
		## Update the weights
        for ix ∈ 1:length(weights)                # loop over each weight
            ## Recall that feature_matrix[:, ix] is the feature column 
		    ## associated with weights[i]
            deriv_ix = feature_derivative_ridge(errors, f_matrix[:, ix],
				weights[ix], 
				l2_penalty; 
				is_constant=ix == 1) 
			
			## subtract the step size times the derivative from the CURRENT weight
            weights[ix] -= η * deriv_ix
		end
	end   
    weights
end

# ╔═╡ 336f9d62-73c0-11eb-1861-3fb709350f2a
md"""
### Visualizing effect of L2 penalty

The L2 penalty gets its name because it causes weights to have small L2 norms than otherwise. Let's see how large weights get penalized. Let us consider a simple model with 1 feature:
"""

# ╔═╡ e8757854-73c9-11eb-2fb9-c10ac86bf9f0
begin
	sales_train, sales_test = train_test_split(sales; shuffled=false);
	first(sales_train, 3)
end

# ╔═╡ e9114fbe-73cd-11eb-052b-15279aaada23
begin
	s_features = [:sqft_living]
	s_output = :price
	
	## use: sales_train, sales_test
	(s_feature_mtr, s_output_tr) = get_data(sales_train, s_features, s_output)
	(s_feature_mte, s_output_te) = get_data(sales_test, s_features, s_output)
	
	s_init_weights = Float64[0., 0.]
	η = 1e-12
	max_iter=1000
	
	s_l2_penalty = 0.0    # no penalty
	h_l2_penalty = 1e11   # high penalty
	
	size(s_feature_mtr), size(s_output_tr), size(s_feature_mte), size(s_output_te)
end

# ╔═╡ 47de5248-73c0-11eb-2a0c-85982a16d3a3
s_weights_0_penalty = regression_gradient_descent(s_feature_mtr,
		Vector{Float64}(s_output_tr), s_init_weights, η, s_l2_penalty;
		max_iter)

# ╔═╡ 5cb2aee6-73c3-11eb-0eb0-cfbfcb4c145a
s_weights_h_penalty = regression_gradient_descent(s_feature_mtr,
		Vector{Float64}(s_output_tr), s_init_weights, η, h_l2_penalty;
		max_iter)

# ╔═╡ c20a95dc-73ce-11eb-072e-cfdf28bb8412
(s_weights_0_penalty, s_weights_h_penalty)

# ╔═╡ 87b80b2c-73c3-11eb-3175-0dedf19b1858
begin
	scatter(s_feature_mtr, s_output_tr, color=[:orange], marker=".")
	plot!(s_feature_mtr, predict_output(s_feature_mtr, s_weights_0_penalty), color=[:blue]) 
	plot!(s_feature_mtr, predict_output(s_feature_mtr, s_weights_h_penalty), color=[:green])
end

# ╔═╡ a2b905e0-73cb-11eb-0aef-79f5e6941f30
md"""
Compute the RSS on the *test* data for the following three sets of weights:
  1. The initial weights (all zeros)
  1. The weights learned with no regularization
  1. The weights learned with high regularization

Which weights perform best?
"""

# ╔═╡ b8f7c40e-73cb-11eb-1e61-8f84439c8cb2
function calc_rss(X, y, weights)
    preds = predict_output(X, weights)
	sum((preds .- y) .^ 2)
end

# ╔═╡ cd05b5fe-73cc-11eb-35a4-952fa79374f4
begin
	## RSS initial weigths
	v0 = calc_rss(s_feature_mte, s_output_te, s_init_weights)
	
	## RSS weigths no regularization
	v1 = calc_rss(s_feature_mte, s_output_te, s_weights_0_penalty) 
	
	## RSS weigths with high regularization
	v2 = calc_rss(s_feature_mte, s_output_te, s_weights_h_penalty) 
	
	with_terminal() do
		@printf("rss init. weights:     %1.5e\n", v0)
		@printf("rss no reg. weights:   %1.5e\n", v1)
		@printf("rss high reg. weights: %1.5e\n", v2)
	end
end

# ╔═╡ 92029980-73cd-11eb-1de5-9fd4c888d352
md"""

**Quiz Questions**

1. What is the value of the coefficient for `sqft_living` that you learned with no regularization, rounded to 1 decimal place?  What about the one with high regularization?
   - 263.0 (no regularization), 68.7 (with high regularization)
  
   
2. Comparing the lines you fit with the with no regularization versus high regularization, which one is steeper?
   - the blue line (with no regularization) 
  
   
3. What are the RSS on the test data for each of the set of weights above (initial, no regularization, high regularization)? 
   - cf. above
"""

# ╔═╡ 27319a50-73d1-11eb-13ee-5b868486e887
md"""
### Running a multiple regression with L2 penalty

Let us now consider a model with 2 features: `[:sqft_living, :sqft_living15]`.

"""

# ╔═╡ 4def015a-73d1-11eb-07be-af419657dd3d
begin	
	m_features = [:sqft_living, :sqft_living15]
	m_output = :price
	
	## use: sales_train, sales_test
	(m_feature_mtr, m_output_tr) = get_data(sales_train, m_features, m_output)
	(m_feature_mte, m_output_te) = get_data(sales_test, m_features, m_output)
	
	m_init_weights = Float64[0., 0., 0.]
	
	## As above for the following:
	# η = 1e-12
	# max_iter=1000
	# s_l2_penalty = 0.0    # no penalty
	# h_l2_penalty = 1e11   # high penalty
end

# ╔═╡ 505bc8cc-73d1-11eb-104b-f1fe8928f397
m_weights_0_penalty = regression_gradient_descent(m_feature_mtr,
		Vector{Float64}(m_output_tr), m_init_weights, η, s_l2_penalty;
		max_iter)

# ╔═╡ 502fc1c8-73d1-11eb-3102-4303736bcc5f
m_weights_h_penalty = regression_gradient_descent(m_feature_mtr,
		Vector{Float64}(m_output_tr), m_init_weights, η, h_l2_penalty;
		max_iter)

# ╔═╡ 4ff2dc02-73d1-11eb-333d-07031c47234b
begin
	## RSS initial weigths
	mv0 = calc_rss(m_feature_mte, m_output_te, m_init_weights)
	
	## RSS weigths no regularization
	mv1 = calc_rss(m_feature_mte, m_output_te, m_weights_0_penalty) 
	
	## RSS weigths with high regularization
	mv2 = calc_rss(m_feature_mte, m_output_te, m_weights_h_penalty) 
	
	with_terminal() do
		@printf("rss init. weights:     %1.5e\n", mv0)
		@printf("rss no reg. weights:   %1.5e\n", mv1)
		@printf("rss high reg. weights: %1.5e\n", mv2)
	end
end

# ╔═╡ 4fdf127a-73d1-11eb-1b41-67a0caaa1eca
begin
	## house price for the 1st house in the test set using the no regularization 
	pred_0_no_reg = predict_output(m_feature_mte, m_weights_0_penalty)[1]
	
	## house price for the 1st house in the test set using high regularization
	pred_0_high_reg = predict_output(m_feature_mte, m_weights_h_penalty)[1]
	
	with_terminal() do
		@printf("no reg. price: %6.2f / price rounded to nearest dollar: %6.2f\n", 
			pred_0_no_reg, round(pred_0_no_reg; digits=0))
	
		@printf("high reg. price: %6.2f / price rounded to nearest dollar: %6.2f\n", 
			pred_0_high_reg, round(pred_0_high_reg; digits=0))
	
		@printf("actual price: %6.2f\n", m_output_te[1])
	end
end

# ╔═╡ 4fb0a88e-73d1-11eb-3eea-339d461b9e17
abs(pred_0_high_reg - m_output_te[1]), abs(pred_0_no_reg - m_output_te[1])

# ╔═╡ 4f8574ac-73d1-11eb-2b52-dd15c43a1c73
md"""
**Quiz Questions**

 1. What is the value of the coefficient for `sqft_living` that you learned with no regularization, rounded to 1 decimal place?  What about the one with high regularization?
   - 189.3 (no reg), 55.2 (high reg)
  

 2. What are the RSS on the test data for each of the set of weights above (initial, no regularization, high regularization)? 
   -  cf. above
   

 3. We make prediction for the first house in the test set using two sets of weights (no regularization vs high regularization). Which weights make better prediction *for that particular house*?
   - price from no reg. model

"""

# ╔═╡ Cell order:
# ╟─9dcffb1c-71c1-11eb-2402-39eaa1f1b5e1
# ╟─c39f9348-71c1-11eb-1691-1d28c6b1d916
# ╠═1bcf25d8-71c2-11eb-05ca-6fe7c4e1b24f
# ╠═1f7ea604-71c2-11eb-028a-a166f6f4dc5d
# ╟─3e5d907e-71c2-11eb-2eb3-4375fb21b884
# ╠═4c736078-71c2-11eb-225b-816745199225
# ╠═0623d920-71ce-11eb-0936-870685f2e788
# ╟─35b5d05a-71cc-11eb-15a9-852c5a534fab
# ╟─a2928dc6-71c2-11eb-3724-29dc1448a0f3
# ╠═e4da992e-71c2-11eb-36b8-93dd614051c1
# ╠═02f0c99e-71c5-11eb-1fe1-b3f820998d54
# ╟─21635b6e-71c8-11eb-2272-edec98c58339
# ╠═96e7042c-73be-11eb-3ac5-9f9998b8c8ff
# ╠═f377f6d4-71c9-11eb-0249-f3d63e87062b
# ╠═59e3f22a-73bd-11eb-1c0d-150d7d462676
# ╟─239ff6ca-71cc-11eb-341a-df1453d60a15
# ╠═958f81a6-71cc-11eb-13e8-61dddeb014a7
# ╟─336f9d62-73c0-11eb-1861-3fb709350f2a
# ╠═e8757854-73c9-11eb-2fb9-c10ac86bf9f0
# ╠═e9114fbe-73cd-11eb-052b-15279aaada23
# ╠═47de5248-73c0-11eb-2a0c-85982a16d3a3
# ╠═5cb2aee6-73c3-11eb-0eb0-cfbfcb4c145a
# ╠═c20a95dc-73ce-11eb-072e-cfdf28bb8412
# ╠═87b80b2c-73c3-11eb-3175-0dedf19b1858
# ╟─a2b905e0-73cb-11eb-0aef-79f5e6941f30
# ╠═b8f7c40e-73cb-11eb-1e61-8f84439c8cb2
# ╠═cd05b5fe-73cc-11eb-35a4-952fa79374f4
# ╟─92029980-73cd-11eb-1de5-9fd4c888d352
# ╟─27319a50-73d1-11eb-13ee-5b868486e887
# ╠═4def015a-73d1-11eb-07be-af419657dd3d
# ╠═505bc8cc-73d1-11eb-104b-f1fe8928f397
# ╠═502fc1c8-73d1-11eb-3102-4303736bcc5f
# ╠═4ff2dc02-73d1-11eb-333d-07031c47234b
# ╠═4fdf127a-73d1-11eb-1b41-67a0caaa1eca
# ╠═4fb0a88e-73d1-11eb-3eea-339d461b9e17
# ╠═4f8574ac-73d1-11eb-2b52-dd15c43a1c73
