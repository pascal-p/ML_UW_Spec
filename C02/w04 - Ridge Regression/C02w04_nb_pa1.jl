### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 5bf16f26-7191-11eb-1166-73a483cf120a
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
end

# ╔═╡ 3cfba9ce-7191-11eb-2295-3b4c85656c03
begin
	using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	using Test
	using Printf
	using Random
 	using Plots  # using PyPlot
end

# ╔═╡ 87aaecb0-7190-11eb-1572-d9110d9ffaaa
md"""
# Week 4: Ridge Regression (Interpretation)
"""

# ╔═╡ c5c92502-7190-11eb-3a75-87b7f156a7ce
md"""
In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
* Use a pre-built implementation of regression (MLJLinearModels) to run polynomial regression, this time with L2 penalty
* Use plot to visualize polynomial regressions under L2 regularization
* Choose best L2 penalty using cross-validation.
* Assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

"""

# ╔═╡ 14d8375a-734e-11eb-2518-2d4eee6b9e3e
md"""
### Polynomial regression, revisited

We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:
"""

# ╔═╡ 55fb5a1e-734e-11eb-2019-21b125a173d1
function polynomial_df(feature; degree=3)
	@assert degree ≥ 1 "Expect degree to be ≥ 1"
	
	hsh = Dict{Symbol, Vector{Float64}}(:power_1 => feature)
	for deg ∈ 2:degree
		hsh[Symbol("power_$(deg)")] = feature .^ deg	
	end
	
	return DataFrame(hsh)
end

# ╔═╡ 7a50bd00-7191-11eb-2ed6-071e6db68422
md"""
### Load in house sales data
"""

# ╔═╡ 890cb362-7191-11eb-251b-e155ed2d21f2
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ b43046d2-7191-11eb-0978-b1bfe1762756
md"""
### Split data into train/test sets
"""

# ╔═╡ cd45cf32-7191-11eb-1e68-53bccaa9d806
function train_test_split(df; split=0.8, seed=42, shuffled=true) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffled ? shuffle(1:nr) : 1:nr
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	(df_train, df_test)
end

# ╔═╡ 76617496-734e-11eb-0177-ef53e0a61ebc
begin
	sort!(sales, [:sqft_living, :price], rev=[false, false]);
	first(sales, 3)
end

# ╔═╡ a626fab6-734e-11eb-0994-f153e8264aaf
md"""
Let us revisit the 15th-order polynomial model using the `sqft_living` input. Generate polynomial features up to degree 15 using `polynomial_df()` and fit a model with these features. 

When fitting the model, use an L2 penalty of 1e-5: 
"""

# ╔═╡ 9f4449e2-734e-11eb-0d61-19d14b53358a
begin
	poly_df_0 = polynomial_df(sales.sqft_living; degree=15)
	poly_df_0[!, :price] = sales.price
	
	first(poly_df_0, 3)
end

# ╔═╡ e985d03e-734e-11eb-26a7-effe0c92f0fc
# @load LinearRegressor pkg=MLJLinearModels
@load RidgeRegressor pkg=MLJLinearModels

# ╔═╡ ed0179fc-734e-11eb-3309-03b0ef4cdc96
begin
	l2_small_penalty = 1e-5
	
	mdl0 = MLJLinearModels.RidgeRegressor(lambda=l2_small_penalty, 
		penalize_intercept=false)
	
	X_0 = select(poly_df_0, :power_1)
	y_0 = poly_df_0.price
	
	mach0 = machine(mdl0, X_0, y_0)
	fit!(mach0)
	fp0 = fitted_params(mach0)
	
	with_terminal() do
		for (name, c) in fp0.coefs
   			println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
		end
		
		println("Intercept: $(round(fp0.intercept, sigdigits=3))")
	end
end

# ╔═╡ f4ac831a-7351-11eb-10b2-67efef23c436
md"""

**Quiz Qestion:  What's the learned value for the coefficient of feature `power_1`?**
  - Answer: 275.0
"""

# ╔═╡ 1fe9cdec-7352-11eb-1336-d98c99d2a002
md"""
### Observe overfitting

Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. <br />

But first, we must reproduce the experiment we did in Week 3.
"""

# ╔═╡ e054e324-7191-11eb-3e15-c1e0bf73bb0d
sales_train, sales_test = train_test_split(sales);

# ╔═╡ 4835efa8-7352-11eb-27dd-23dcf67ead55
begin
	(ssales_a, ssales_b) = train_test_split(sales; split=0.5, seed=42) 

	(set1, set2) = train_test_split(ssales_a; split=0.5, seed=42)
	(set3, set4) = train_test_split(ssales_b; split=0.5, seed=42)

	(size(set1), size(set2), size(set3), size(set4))
end

# ╔═╡ b3b02ed6-7352-11eb-3a15-9f392deb56d8
begin

function make_poly(tset; degree=15, output=:price)
  poly_df = polynomial_df(tset.sqft_living; degree) 
  features = names(poly_df)   
  poly_df[!, output] = tset[!, output]
  (features, poly_df)
end

function fit_poly(tset; degree=15, output=:price, l2_penalty=l2_small_penalty)
  (features, poly_df) = make_poly(tset; degree, output) 
  mdl = MLJLinearModels.RidgeRegressor(lambda=l2_penalty, 
		penalize_intercept=false)	
  X_ = select(poly_df, features)
  y_ = poly_df[!, output]
	
  mach = machine(mdl, X_, y_)
  fit!(mach)
 
  (mach, X_, y_)
end
	
function print_coeff(mach)
  fp = fitted_params(mach)
	
  with_terminal() do
	for (name, c) in fp.coefs[1:1]  # print only first coeff
	  println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
    end
		
	println("Intercept: $(round(fp.intercept, sigdigits=3))")
  end
end
	
end

# ╔═╡ 6431250e-7354-11eb-1446-37f9f51cb959
md"""
##### Set1
"""

# ╔═╡ 64f0b460-7353-11eb-23b5-0b2a0c9006f4
## set 1
begin
	(mach_set1, Xset1, yset1) = fit_poly(set1)
	print_coeff(mach_set1)
end

# ╔═╡ 3bf25c7a-7354-11eb-335e-73edbc852a30
## Visualization
begin
	# figure(figsize=(8,6))
	# plot(Xset1.power_1, yset1, color=[:lightblue], marker=".")
	scatter(Xset1.power_1, yset1, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset1.power_1, predict(mach_set1, Xset1), color=[:orange], marker="-")
	# plot(Xset1.power_1, predict(mach_set1, Xset1), color=[:orange]) #, marker="-")
end

# ╔═╡ 84bc0834-7354-11eb-30bb-e9532e430f6f
md"""
##### Set2
"""

# ╔═╡ 521d7282-7354-11eb-0b52-e7448ad1b4e6
begin
	(mach_set2, Xset2, yset2) = fit_poly(set2)
	print_coeff(mach_set2)
end

# ╔═╡ 8f6ece26-7354-11eb-30ad-8976c7749a18
begin
	scatter(Xset2.power_1, yset2, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset2.power_1, predict(mach_set2, Xset2), color=[:orange], marker="-")
end

# ╔═╡ a10ed532-7354-11eb-3013-3f50c4bb610d
md"""
##### set3
"""

# ╔═╡ a0e22660-7354-11eb-33a7-49f30c590f07
begin
	(mach_set3, Xset3, yset3) = fit_poly(set3)
	print_coeff(mach_set3)
end

# ╔═╡ a09f0c10-7354-11eb-3906-4176b483705e
begin
	scatter(Xset3.power_1, yset3, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset3.power_1, predict(mach_set3, Xset3), color=[:orange], marker="-")
end

# ╔═╡ 9f660178-7354-11eb-213c-5fa979a85749
md"""
##### set4
"""

# ╔═╡ 0483452a-71a0-11eb-30c0-7d56ed83e837
begin
	(mach_set4, Xset4, yset4) = fit_poly(set4)
	print_coeff(mach_set4)
end

# ╔═╡ e05ea0ae-7354-11eb-30fc-65340e99f59c
begin
	scatter(Xset4.power_1, yset4, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset4.power_1, predict(mach_set4, Xset4), color=[:orange], marker="-")
end

# ╔═╡ e4c6e384-7354-11eb-307d-579569191332
md"""
The four curves should differ from one another a lot, as should the coefficients you learned.


**Quiz Question:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?**  

  - smallest: -536.0 (set3)
  - largest: 748.0 (set1)
"""

# ╔═╡ 40bd2ee6-7355-11eb-00ba-1943adaa3fb9
md"""
### Ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set1`, `set2`, `set3`, and `set4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above.
"""

# ╔═╡ 5aa4d1cc-7355-11eb-09a9-dd185f7871e1
l2_penalty=1e5

# ╔═╡ 5aa44ab0-7355-11eb-17cb-7bd692a55a37
md"""
#### Set1
"""

# ╔═╡ 5a6ddeb2-7355-11eb-2789-ab29446d2fd5
# set 1
begin
	(mach_set1b, Xset1b, yset1b) = fit_poly(set1; l2_penalty)
	print_coeff(mach_set1b)
end

# ╔═╡ 5a57a7aa-7355-11eb-311a-49ee4c23fa4b
## Visualization
begin
	scatter(Xset1b.power_1, yset1b, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset1b.power_1, predict(mach_set1b, Xset1b), color=[:orange], marker="-")
end

# ╔═╡ 5a40f352-7355-11eb-0e45-99b96319953e
md"""
#### Set 2
"""

# ╔═╡ 5a1059b8-7355-11eb-219c-3bde9ab3be34
begin
	(mach_set2b, Xset2b, yset2b) = fit_poly(set2; l2_penalty)
	print_coeff(mach_set2b)
end

# ╔═╡ 59f91dfc-7355-11eb-2e4d-d1f62739cbee
begin
	scatter(Xset2b.power_1, yset2b, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset2b.power_1, predict(mach_set2b, Xset2b), color=[:orange], marker="-")
end

# ╔═╡ 59b7c226-7355-11eb-3061-d574d86ef5be
md"""
#### Set 3
"""

# ╔═╡ d1b6de72-7355-11eb-1f7d-2f2e3d830461
begin
	(mach_set3b, Xset3b, yset3b) = fit_poly(set3; l2_penalty)
	print_coeff(mach_set3b)
end

# ╔═╡ 5948656e-7355-11eb-28cc-ef04179d0281
begin
	scatter(Xset3b.power_1, yset3b, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset3b.power_1, predict(mach_set3b, Xset3b), color=[:orange], marker="-")
end

# ╔═╡ d4268fa6-7355-11eb-09dd-0f98bbf606f3
md"""
#### Set 4
"""

# ╔═╡ d3f53028-7355-11eb-1a5a-2545b7bfb1d8
begin
	(mach_set4b, Xset4b, yset4b) = fit_poly(set4; l2_penalty)
	print_coeff(mach_set4b)
end

# ╔═╡ d399ba7c-7355-11eb-0360-d940e85424c5
begin
	scatter(Xset4b.power_1, yset4b, legend=false, color=[:lightblue], marker=".")
 	scatter!(Xset4b.power_1, predict(mach_set4b, Xset4b), color=[:orange], marker="-")
end

# ╔═╡ d32b7d62-7355-11eb-00f5-e15ead474ce5
md"""
These curves should vary a lot less, now that you applied a high degree of regularization.


**Quiz Question:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?** 

  - smallest: 263.0 (set3)
  - largest: 597.0 (set14
"""

# ╔═╡ 8e9ed252-7357-11eb-117c-91f7ff585a2d
md"""
### Selecting an L2 penalty via cross-validation


Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
...

Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. 

"""

# ╔═╡ 91a058b6-7359-11eb-1da5-eba5d339d100
function get_rss(mach, X, y)
    ŷ = predict(mach, X)     # First get the predictions
    diff = y .- ŷ            # Then compute the residuals/errors
    rss = sum(diff .* diff)  # Then square and add them up
    return rss
end

# ╔═╡ 957b90fe-7359-11eb-23fc-039055cd2914
function k_fold_cross_validation(k, l2_penalty, df, output, features; 
	degree=15)
	n = size(df)[1]
	rss = []
	for ix ∈ 1:k
		start_ = (n * (ix - 1)) ÷ k + 1
		end_ = (n * ix) ÷ k
		
		## get the partition train/valid
		valid_set = df[start_:end_, :]
		train_set = vcat(df[1:start_ - 1, :], df[end_+1:end, :])
		
		##
		mdl = MLJLinearModels.RidgeRegressor(lambda=l2_penalty,  
			penalize_intercept=false)	
  		X_ = select(train_set, features)
  		y_ = train_set[!, output]
	
  		mach = machine(mdl, X_, y_)
  		fit!(mach)
	
		##
		push!(rss, get_rss(mach, valid_set, valid_set[!, output]))
	end
	sum(rss) / k
end

# ╔═╡ 9af5697a-73af-11eb-1558-dbc635b07638
function find_best_l2(train_valid; k=10, degree=15, output=:price)
	l2_rss, l2_p = [], []
	best_l2, min_rss = nothing, nothing
	(features, poly_set) = make_poly(train_valid; degree, output)
	
	for cur_l2p ∈ (10^ix for ix ∈ 1:0.5:7)
		push!(l2_p, cur_l2p)
		rss = k_fold_cross_validation(k, cur_l2p, poly_set, output, features)
		push!(l2_rss, rss)
		
		# @printf("current rss: %2.5e / current penalty: %2.5f\n", rss, cur_l2p)
		if isnothing(min_rss) || rss < min_rss
			min_rss = rss
			best_l2 = cur_l2p
		end
	end
	(best_l2, min_rss, l2_p, l2_rss)
end

# ╔═╡ 51150d2e-7358-11eb-2d17-c5c6898b7045
begin
	# train_valid is shuffled during the train-test_split
	(train_valid, test) = train_test_split(sales; split=0.9, seed=42)
	
	k = 10
	degree = 15
	output=:price
	
	(best_l2, min_rss, l2_p, l2_rss) = find_best_l2(train_valid; k, degree, output)
	with_terminal() do
		@printf("min rss: %2.5e / best penalty: %2.5e\n", min_rss, best_l2)
	end
end

# ╔═╡ daa7efec-73b0-11eb-3657-d520436222d5
plot(l2_p, l2_rss, marker="-", color=[:lightblue], xscale=:log)

# ╔═╡ b273d2d6-73b1-11eb-06df-1559778e561d
md"""
Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of l2_penalty.


This way, your final model will be trained on the entire dataset.
"""

# ╔═╡ 199f8240-73b2-11eb-2753-ab74e806ab9f
begin
	(features, train_set) = make_poly(train_valid; degree, output)
	
	mdl = MLJLinearModels.RidgeRegressor(lambda=best_l2,  
			penalize_intercept=false)	
  		X_ = select(train_set, features)
  		y_ = train_set[!, output]
	
  		mach = machine(mdl, X_, y_)
  		fit!(mach)
	
	(_, test_set) = make_poly(test; degree, output)
	
	final_rss = get_rss(mach, test_set, test_set[!, output])
	
	with_terminal() do
		@printf("final rss: %2.5e\n", final_rss)
	end
end

# ╔═╡ 3db30566-73b3-11eb-1d4e-f919ad63641a
md"""
**Quiz Question: Using the best L2 penalty found above, train a model using all *training* data. What is the RSS on the *test* data of the model you learn with this L2 penalty?**

 - Final rss: 4.03826e+14
"""

# ╔═╡ Cell order:
# ╟─87aaecb0-7190-11eb-1572-d9110d9ffaaa
# ╟─c5c92502-7190-11eb-3a75-87b7f156a7ce
# ╠═5bf16f26-7191-11eb-1166-73a483cf120a
# ╠═3cfba9ce-7191-11eb-2295-3b4c85656c03
# ╟─14d8375a-734e-11eb-2518-2d4eee6b9e3e
# ╠═55fb5a1e-734e-11eb-2019-21b125a173d1
# ╟─7a50bd00-7191-11eb-2ed6-071e6db68422
# ╠═890cb362-7191-11eb-251b-e155ed2d21f2
# ╟─b43046d2-7191-11eb-0978-b1bfe1762756
# ╠═cd45cf32-7191-11eb-1e68-53bccaa9d806
# ╠═76617496-734e-11eb-0177-ef53e0a61ebc
# ╟─a626fab6-734e-11eb-0994-f153e8264aaf
# ╠═9f4449e2-734e-11eb-0d61-19d14b53358a
# ╠═e985d03e-734e-11eb-26a7-effe0c92f0fc
# ╠═ed0179fc-734e-11eb-3309-03b0ef4cdc96
# ╟─f4ac831a-7351-11eb-10b2-67efef23c436
# ╟─1fe9cdec-7352-11eb-1336-d98c99d2a002
# ╠═e054e324-7191-11eb-3e15-c1e0bf73bb0d
# ╠═4835efa8-7352-11eb-27dd-23dcf67ead55
# ╠═b3b02ed6-7352-11eb-3a15-9f392deb56d8
# ╟─6431250e-7354-11eb-1446-37f9f51cb959
# ╠═64f0b460-7353-11eb-23b5-0b2a0c9006f4
# ╠═3bf25c7a-7354-11eb-335e-73edbc852a30
# ╟─84bc0834-7354-11eb-30bb-e9532e430f6f
# ╠═521d7282-7354-11eb-0b52-e7448ad1b4e6
# ╠═8f6ece26-7354-11eb-30ad-8976c7749a18
# ╟─a10ed532-7354-11eb-3013-3f50c4bb610d
# ╠═a0e22660-7354-11eb-33a7-49f30c590f07
# ╠═a09f0c10-7354-11eb-3906-4176b483705e
# ╟─9f660178-7354-11eb-213c-5fa979a85749
# ╠═0483452a-71a0-11eb-30c0-7d56ed83e837
# ╠═e05ea0ae-7354-11eb-30fc-65340e99f59c
# ╟─e4c6e384-7354-11eb-307d-579569191332
# ╟─40bd2ee6-7355-11eb-00ba-1943adaa3fb9
# ╠═5aa4d1cc-7355-11eb-09a9-dd185f7871e1
# ╟─5aa44ab0-7355-11eb-17cb-7bd692a55a37
# ╠═5a6ddeb2-7355-11eb-2789-ab29446d2fd5
# ╠═5a57a7aa-7355-11eb-311a-49ee4c23fa4b
# ╟─5a40f352-7355-11eb-0e45-99b96319953e
# ╠═5a1059b8-7355-11eb-219c-3bde9ab3be34
# ╠═59f91dfc-7355-11eb-2e4d-d1f62739cbee
# ╟─59b7c226-7355-11eb-3061-d574d86ef5be
# ╠═d1b6de72-7355-11eb-1f7d-2f2e3d830461
# ╠═5948656e-7355-11eb-28cc-ef04179d0281
# ╟─d4268fa6-7355-11eb-09dd-0f98bbf606f3
# ╠═d3f53028-7355-11eb-1a5a-2545b7bfb1d8
# ╠═d399ba7c-7355-11eb-0360-d940e85424c5
# ╟─d32b7d62-7355-11eb-00f5-e15ead474ce5
# ╟─8e9ed252-7357-11eb-117c-91f7ff585a2d
# ╠═91a058b6-7359-11eb-1da5-eba5d339d100
# ╠═957b90fe-7359-11eb-23fc-039055cd2914
# ╠═9af5697a-73af-11eb-1558-dbc635b07638
# ╠═51150d2e-7358-11eb-2d17-c5c6898b7045
# ╠═daa7efec-73b0-11eb-3657-d520436222d5
# ╟─b273d2d6-73b1-11eb-06df-1559778e561d
# ╠═199f8240-73b2-11eb-2753-ab74e806ab9f
# ╟─3db30566-73b3-11eb-1d4e-f919ad63641a
