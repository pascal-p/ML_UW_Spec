### A Pluto.jl notebook ###
# v0.12.21

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
sort!(sales, [:sqft_living, :price], rev=[false, false]);

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
  - Answer below:
"""

# ╔═╡ 4686d706-74a8-11eb-101d-c976b98872f3
round(fp0.coefs[1][2], sigdigits=2)

# ╔═╡ 3c18cc2e-74a9-11eb-336b-1bdf4159d5f8
getfield(fp0, :coefs)[1][2]

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
	
function print_coeff(mach, ix)
  fp = fitted_params(mach)	
  println("\n- model $(ix)")
  for (name, c) in fp.coefs[1:1]  # print only first coeff
	println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
  end
		
  println("Intercept: $(round(fp.intercept, sigdigits=3))")
end
	
function find(machs; key=:min)
	fps = map(m -> fitted_params(m), machs) |>
		fps -> map(fp -> getfield(fp, :coefs)[1][2], fps)
		
	ix = key == :min ? argmin(fps) : argmax(fps)
	(fps[ix], string("model_", ix))
end
	
end

# ╔═╡ 6431250e-7354-11eb-1446-37f9f51cb959
md"""
##### Set1, Set2, Set3 & Set4
"""

# ╔═╡ 64f0b460-7353-11eb-23b5-0b2a0c9006f4
begin
	hsh = Dict{Symbol, Vector{Any}}(:machine => [], :tset => [])
	
	for (ix, s) ∈ enumerate((set1, set2, set3, set4))
		(mach_, X_, y_) = fit_poly(s)
		push!(hsh[:machine], mach_)
		push!(hsh[:tset], (X_, y_))
	end
	
	ps = []

	for ix ∈ 1:length(hsh[:machine])
		(X_, y_) = hsh[:tset][ix]
		mach_ = hsh[:machine][ix]
		
		p1 = scatter(X_.power_1, y_, legend=false, color=[:lightblue], marker=".")
 		p2 = scatter!(X_.power_1, predict(mach_, X_), color=[:orange], marker="-")
		push!(ps, (p1, p2))
	end
	
end

# ╔═╡ c7c9f6dc-7495-11eb-2bd5-8ba7f20f997a
with_terminal() do
	for ix in 1:length(hsh[:machine])
		print_coeff(hsh[:machine][ix], ix)
	end
end

# ╔═╡ e22363b4-7499-11eb-3b50-5905e5eef045
begin
	lg1 = grid(2, 2, widths=[0.0, 0.9], heights=[0.5, 0.5])	
	plot(ps[1]..., ps[2]..., layout=lg1)
end

# ╔═╡ f08611a8-749a-11eb-1014-e9e29e34a966
begin
	lg2 = grid(2, 2, widths=[0.0, 0.9], heights=[0.5, 0.5])
	plot(ps[3]..., ps[4]..., layout=lg2)
end

# ╔═╡ e4c6e384-7354-11eb-307d-579569191332
md"""
The four curves should differ from one another a lot, as should the coefficients you learned.


**Quiz Question:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?**  

  - cf. below
"""

# ╔═╡ ac0d2d96-74a8-11eb-079e-036266a3b75e
(smallest=find(hsh[:machine]), largest=find(hsh[:machine]; key=:max))

# ╔═╡ 40bd2ee6-7355-11eb-00ba-1943adaa3fb9
md"""
### Ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set1`, `set2`, `set3`, and `set4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above.
"""

# ╔═╡ 5aa44ab0-7355-11eb-17cb-7bd692a55a37
md"""
#### Set1, Set2, Set3 & Set4
"""

# ╔═╡ 5aa4d1cc-7355-11eb-09a9-dd185f7871e1
l2_penalty=1e5

# ╔═╡ 96c23cd0-749c-11eb-0c80-63c4f283c2e5
begin
	hsh_v1r = Dict{Symbol, Vector{Any}}(:machine => [], :tset => [])
	
	for (ix, s) ∈ enumerate((set1, set2, set3, set4))
		(mach_, X_, y_) = fit_poly(s; l2_penalty)
		push!(hsh_v1r[:machine], mach_)
		push!(hsh_v1r[:tset], (X_, y_))
	end
	
	## plot prep.
	ps2 = []
	for ix ∈ 1:length(hsh[:machine])
		(X_, y_) = hsh[:tset][ix]
		mach_ = hsh[:machine][ix]
		
		p1 = scatter(X_.power_1, y_, legend=false, color=[:lightblue], marker=".")
 		p2 = scatter!(X_.power_1, predict(mach_, X_), color=[:orange], marker="-")
		push!(ps2, (p1, p2)) # plot(p1, p2, layout=l)
	end
	
	with_terminal() do
		for ix in 1:length(hsh_v1r[:machine])
			print_coeff(hsh_v1r[:machine][ix], ix)
		end
	end
end

# ╔═╡ 03e1ed82-74a8-11eb-140c-e31183a2314b
begin
	lg11 = grid(2, 2, widths=[0.0, 0.9], heights=[0.5, 0.5])	
	plot(ps2[1]..., ps2[2]..., layout=lg11)
end

# ╔═╡ 034e5e96-74a8-11eb-030d-fb9f7ea9af55
begin
	lg12 = grid(2, 2, widths=[0.0, 0.9], heights=[0.5, 0.5])
	plot(ps2[3]..., ps2[4]..., layout=lg12)
end

# ╔═╡ d32b7d62-7355-11eb-00f5-e15ead474ce5
md"""
These curves should vary a lot less, now that you applied a high degree of regularization.


**Quiz Question:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?** 

  - cf. below
"""

# ╔═╡ a96763a2-74aa-11eb-2a92-87ee73d05221
(smallest=find(hsh_v1r[:machine]), largest=find(hsh_v1r[:machine]; key=:max))

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
# ╠═4686d706-74a8-11eb-101d-c976b98872f3
# ╠═3c18cc2e-74a9-11eb-336b-1bdf4159d5f8
# ╟─1fe9cdec-7352-11eb-1336-d98c99d2a002
# ╠═e054e324-7191-11eb-3e15-c1e0bf73bb0d
# ╠═4835efa8-7352-11eb-27dd-23dcf67ead55
# ╠═b3b02ed6-7352-11eb-3a15-9f392deb56d8
# ╟─6431250e-7354-11eb-1446-37f9f51cb959
# ╠═64f0b460-7353-11eb-23b5-0b2a0c9006f4
# ╠═c7c9f6dc-7495-11eb-2bd5-8ba7f20f997a
# ╠═e22363b4-7499-11eb-3b50-5905e5eef045
# ╠═f08611a8-749a-11eb-1014-e9e29e34a966
# ╟─e4c6e384-7354-11eb-307d-579569191332
# ╠═ac0d2d96-74a8-11eb-079e-036266a3b75e
# ╟─40bd2ee6-7355-11eb-00ba-1943adaa3fb9
# ╟─5aa44ab0-7355-11eb-17cb-7bd692a55a37
# ╠═5aa4d1cc-7355-11eb-09a9-dd185f7871e1
# ╠═96c23cd0-749c-11eb-0c80-63c4f283c2e5
# ╠═03e1ed82-74a8-11eb-140c-e31183a2314b
# ╠═034e5e96-74a8-11eb-030d-fb9f7ea9af55
# ╟─d32b7d62-7355-11eb-00f5-e15ead474ce5
# ╠═a96763a2-74aa-11eb-2a92-87ee73d05221
# ╟─8e9ed252-7357-11eb-117c-91f7ff585a2d
# ╠═91a058b6-7359-11eb-1da5-eba5d339d100
# ╠═957b90fe-7359-11eb-23fc-039055cd2914
# ╠═9af5697a-73af-11eb-1558-dbc635b07638
# ╠═51150d2e-7358-11eb-2d17-c5c6898b7045
# ╠═daa7efec-73b0-11eb-3657-d520436222d5
# ╟─b273d2d6-73b1-11eb-06df-1559778e561d
# ╠═199f8240-73b2-11eb-2753-ab74e806ab9f
# ╟─3db30566-73b3-11eb-1d4e-f919ad63641a
