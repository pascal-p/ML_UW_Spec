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
end

# ╔═╡ f3283b68-7191-11eb-3290-33ba4d2aaece
using Random

# ╔═╡ 9dc7a434-7195-11eb-35a1-e9f79dd4ee79
using Test

# ╔═╡ 428fadd2-719a-11eb-3f4b-d98cb77cddbe
using Printf

# ╔═╡ 87aaecb0-7190-11eb-1572-d9110d9ffaaa
md"""
# Week 2: Multiple Regression (Interpretation)
"""

# ╔═╡ c5c92502-7190-11eb-3a75-87b7f156a7ce
md"""
The goal of this first notebook is to explore multiple regression and feature engineering with MLJ functions.

In this notebook we will use data on house sales in King County to predict prices using multiple regression. we will:

  - Use DataFrames to do some feature engineering
  - Use built-in MLJ functions to compute the regression weights (coefficients/parameters)
  - Given the regression weights, predictors and outcome write a function to compute the Residual Sum of Squares (RSS)
  - Look at coefficients and interpret their meanings
  - Evaluate multiple models via RSS

"""

# ╔═╡ 7a50bd00-7191-11eb-2ed6-071e6db68422
md"""
### Load in house sales data
"""

# ╔═╡ 890cb362-7191-11eb-251b-e155ed2d21f2
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ b1ae4466-7191-11eb-0c07-29e30cce116a
first(sales, 1)

# ╔═╡ b43046d2-7191-11eb-0978-b1bfe1762756
md"""
### Split data into train/test sets
"""

# ╔═╡ cd45cf32-7191-11eb-1e68-53bccaa9d806
function train_test_split(df; split=0.8, seed=42) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffle(1:nr)
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	(df_train, df_test)
end

# ╔═╡ e054e324-7191-11eb-3e15-c1e0bf73bb0d
sales_train, sales_test = train_test_split(sales);

# ╔═╡ fa6c9bf8-7191-11eb-26b3-1f7f13b91971
size(sales_train), size(sales_test)

# ╔═╡ 0d2ca940-7192-11eb-34ea-ab594b994f41
md"""
### Learning a multiple regression model
"""

# ╔═╡ 14e8024c-7192-11eb-1c76-83a28b6120ee
@load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 72a34b30-7192-11eb-3cda-15a4e9a5244c
describe(sales, :mean, :std, :eltype)

# ╔═╡ c264c3ec-7192-11eb-24ef-c797c42d3671
begin
	example_features = [:sqft_living, :bedrooms, :bathrooms]
	
	X = select(sales_train, example_features)
	y = sales_train.price
	
	first(X, 3)
end

# ╔═╡ 1f6d0112-7193-11eb-20a2-efd1550ca8e2
## 3 first house prices
y[1:3]

# ╔═╡ bd39bf8e-7193-11eb-23c0-257408e760a0
size(X), size(y)

# ╔═╡ 45479604-7193-11eb-068e-7ddd263e9be2
## Let's declare a simple multivariate linear regression model:
mdl = MLJLinearModels.LinearRegressor()


# ╔═╡ 89ce6cb2-7193-11eb-25bb-f161f4514a6d
## Let's wrap it in a machine which, in MLJ, is the composition of a model 
## and data to apply the model on:

begin
	mach_mr = machine(mdl, X, y)
	fit!(mach_mr)
end

# ╔═╡ d34ab594-7193-11eb-1bed-01a763445a7e
begin
	fp = fitted_params(mach_mr)
	
	with_terminal() do
		for (name, val) in fp.coefs
    		println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
		end
		println("Intercept: $(round(fp.intercept, sigdigits=3))")
	end
end

# ╔═╡ 508db812-7194-11eb-3fde-ad654ba107d1
md"""
### Making Predictions
"""

# ╔═╡ 6732d8e0-7194-11eb-3506-93329a94dd84
begin
	ŷ = predict(mach_mr, X)
	(round(ŷ[1], sigdigits=3) , y[1])  # first house predicetd price vs actual price
end

# ╔═╡ c9a0a836-7194-11eb-027a-7bfbfe6cae3c
md"""
### Compute RSS

Now that we can make predictions given the model, let's write a function to compute the RSS [Residual Sum of Squares] of the model. 

Let's write a function `get_rss`  to calculate RSS given the machine, data(X), and the outcome (y).
"""

# ╔═╡ 3deaf822-7195-11eb-2bfb-9f51b24982ed
function get_rss(mach, X, y)
    ŷ = predict(mach, X)     # First get the predictions
    diff = y .- ŷ            # Then compute the residuals/errors
    rss = sum(diff .* diff)  # Then square and add them up
    return rss
end

# ╔═╡ a5251374-7195-11eb-368e-5d4814c8a5e2
begin
	X_test = select(sales_test, example_features)
	y_test = sales_test.price
	
	rss_ex = get_rss(mach_mr, X_test, y_test)
	exp_value = 5.313821155174205e13  # should be 5.313821155174205e13
	@test rss_ex ≈ exp_value
    # println("rss_ex is: $(rss_ex)")
end

# ╔═╡ 83c35d98-7196-11eb-0a75-b952509c4632
md"""
### Create some new features

Let's consider transformations of existing features e.g. the log of the squarefeet or even "interaction" features such as the product of bedrooms and bathrooms.

Create the following 4 new features as column in both test and train data:

  - `bedrooms_squared = bedrooms * bedrooms`
  - `bed_bath_rooms = bedrooms * bathrooms`
  - `log_sqft_living = log(sqft_living)`
  - `lat_plus_long = lat + long` 

"""

# ╔═╡ d1cc2b4a-7197-11eb-10fd-8fdadba255d2
begin
	Xtr, Xte = (sales_train, sales_test)
end

# ╔═╡ d0d2312a-7196-11eb-3192-2158f9b7af15
begin
 	nXtr = hcat(Xtr, 
		Xtr.bedrooms .* Xtr.bedrooms, 
		Xtr.bedrooms .* Xtr.bathrooms,	
 		log.(Xtr.sqft_living), 
		Xtr.lat .+ Xtr.long; 
		makeunique=true);
	
	rename!(nXtr, :x1 => :bedrooms_squared);
	rename!(nXtr, :x1_1 => :bed_bath_rooms);
	rename!(nXtr, :x1_2 => :log_sqft_living);
	rename!(nXtr, :x1_3 => :lat_plus_long);
end

# ╔═╡ bbe4cfb8-7199-11eb-2c60-edaa2feab075
begin
	nXte = hcat(Xte, Xte.bedrooms .* Xte.bedrooms, 
		Xte.bedrooms .* Xte.bathrooms, 	
		log.(Xte.sqft_living), 
		Xte.lat .+ Xte.long;
		makeunique=true);
	
	rename!(nXte, :x1 => :bedrooms_squared);
	rename!(nXte, :x1_1 => :bed_bath_rooms);
	rename!(nXte, :x1_2 => :log_sqft_living);
	rename!(nXte, :x1_3 => :lat_plus_long);
end

# ╔═╡ fd7ca6fa-7199-11eb-0691-89ea26b44013
md"""

**Quiz Question: What is the mean (arithmetic average) value of your 4 new features on 
test data? (round to 2 digits)**

"""

# ╔═╡ 0ac9bda2-719a-11eb-3036-d110c21a2fcc
with_terminal() do
	for attr ∈ [:bed_bath_rooms, :log_sqft_living, :lat_plus_long, :bedrooms_squared]
		@printf("mean of %s is: %3.2f\n", String(attr), mean(nXte[!, attr]))
	end
end

# ╔═╡ afde3d9c-719a-11eb-12cb-8397b9441a65
md"""
### Learning Multiple Models

Now we will learn the weights for three (nested) models for predicting house prices. The first model will have the fewest features the second model will add one more feature and the third will add a few more:
  - Model 1: `squarefeet`, `# bedrooms`, `# bathrooms`, `latitude & longitude`
  - Model 2: add `bedrooms *bathrooms`
  - Model 3: Add `log squarefeet`, `bedrooms squared`, and the (nonsensical) `latitude + longitude`
"""

# ╔═╡ fa0073b6-719a-11eb-3190-dd61c2541ae4
begin
	mod1_features = [:sqft_living, :bedrooms, :bathrooms, :lat, :long];
	mod2_features = [mod1_features..., :bed_bath_rooms];
	mod3_features = [mod2_features..., :bedrooms_squared, :log_sqft_living, :lat_plus_long];
end

# ╔═╡ 5caf09e6-719b-11eb-1652-8938ad992ea3
## model 1
begin
	X1 = select(nXtr, mod1_features)
	mach_mr1 = machine(mdl, X1, y)
	fit!(mach_mr1)
	
	fp1 = fitted_params(mach_mr1)
	
	with_terminal() do
		for (name, val) in fp1.coefs
    		println("$(rpad(name, 16)):  $(round(val, sigdigits=3))")
		end
		println("Intercept: $(round(fp1.intercept, sigdigits=3))")
	end
end

# ╔═╡ c56e31be-719b-11eb-022d-f796b95b6b32
## model 2
begin
	X2 = select(nXtr, mod2_features)
	mach_mr2 = machine(mdl, X2, y)
	fit!(mach_mr2)
	
	fp2 = fitted_params(mach_mr2)
	
	with_terminal() do
		for (name, val) in fp2.coefs
    		println("$(rpad(name, 16)):  $(round(val, sigdigits=3))")
		end
		println("Intercept: $(round(fp2.intercept, sigdigits=3))")
	end
end

# ╔═╡ d972ae42-719b-11eb-193e-675d543025c1
## model 3
begin
	X3 = select(nXtr, mod3_features)
	mach_mr3 = machine(mdl, X3, y)
	fit!(mach_mr3)
	
	fp3 = fitted_params(mach_mr3)
	
	with_terminal() do
		for (name, val) in fp3.coefs
    		println("$(rpad(name, 16)):  $(round(val, sigdigits=3))")
		end
		println("Intercept: $(round(fp3.intercept, sigdigits=3))")
	end
end

# ╔═╡ 266c0a0e-719c-11eb-1692-f74b2a6f5bc4
md"""

**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 1?**
  - [x] Positive (+)
  - [ ] Negative (-)


**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 2?**
  - [ ] Positive (+)
  - [x] Negative (-)


Think about what this means.

"""

# ╔═╡ 2dfa4aee-719c-11eb-1f89-1babfa6312d2
md"""
### Comparing multiple models

Now that we have learned three models and extracted the model weights we want to evaluate which model is best.

"""

# ╔═╡ 48377010-719c-11eb-19a7-233bf9bb4beb
# function calc_rss(;models=[(:model_1, mach_mr1), 
# 			(:model_2, mach_mr2), 
# 			(:model_3, mach_mr3)], tdata=sales_train)
# 	for (modn, mach) ∈ models
#     	rss_mod = get_rss(mach, tdata, tdata.price)
#     	@printf("model %10s rss is: %2.7e", String(modn), rss_mod)
# 	end
#   	return
# end

# ╔═╡ 9af9983a-719e-11eb-0a18-f5493f3df6a3
md"""
**Training Data**
"""

# ╔═╡ 06cdc690-719e-11eb-0819-1b5fce1e3a18
begin
	rss_mmr1 = get_rss(mach_mr1, X1, y)
	rss_mmr2 = get_rss(mach_mr2, X2, y)
	rss_mmr3 = get_rss(mach_mr3, X3, y)
	
	with_terminal() do
		for (ix, rss_) ∈ enumerate((rss_mmr1, rss_mmr2, rss_mmr3))
			@printf("model %2d rss is: %2.7e\n", ix, rss_)
		end
	end	
end

# ╔═╡ e60ea370-719f-11eb-3d56-5dcfecf4ab3e
md"""
**Quiz Question: Which model (1, 2 or 3) has lowest RSS on *training* Data?** Is this what you expected?

  - [ ] Model 1
  - [ ] Model 2
  - [x] Model 3
  
  
  I would expect this (on training data)
"""

# ╔═╡ 90a5c372-719e-11eb-3483-6140d252b0c0
md"""
**Test Data**
"""

# ╔═╡ b2d5046c-719e-11eb-07db-454247474942
begin
	rss_mmr1_te = get_rss(mach_mr1, select(nXte, mod1_features), sales_test.price)
	rss_mmr2_te = get_rss(mach_mr2, select(nXte, mod2_features), sales_test.price)
	rss_mmr3_te = get_rss(mach_mr3, select(nXte, mod3_features), sales_test.price)
	
	with_terminal() do
		for (ix, rss_) ∈ enumerate((rss_mmr1_te, rss_mmr2_te, rss_mmr3_te))
			@printf("model %2d rss is: %2.7e\n", ix, rss_)
		end
	end
end

# ╔═╡ fb4038a6-719f-11eb-3e40-cd5aecfb149b
md"""

**Quiz Question: Which model (1, 2 or 3) has lowest RSS on *testing* Data?** Is this what you expected? Think about the features that were added to each model from the previous.

  - [x] Model 1
  - [ ] Model 2
  - [ ] Model 3

"""

# ╔═╡ 0483452a-71a0-11eb-30c0-7d56ed83e837


# ╔═╡ Cell order:
# ╟─87aaecb0-7190-11eb-1572-d9110d9ffaaa
# ╟─c5c92502-7190-11eb-3a75-87b7f156a7ce
# ╠═5bf16f26-7191-11eb-1166-73a483cf120a
# ╠═3cfba9ce-7191-11eb-2295-3b4c85656c03
# ╟─7a50bd00-7191-11eb-2ed6-071e6db68422
# ╠═890cb362-7191-11eb-251b-e155ed2d21f2
# ╠═b1ae4466-7191-11eb-0c07-29e30cce116a
# ╟─b43046d2-7191-11eb-0978-b1bfe1762756
# ╠═f3283b68-7191-11eb-3290-33ba4d2aaece
# ╠═cd45cf32-7191-11eb-1e68-53bccaa9d806
# ╠═e054e324-7191-11eb-3e15-c1e0bf73bb0d
# ╠═fa6c9bf8-7191-11eb-26b3-1f7f13b91971
# ╟─0d2ca940-7192-11eb-34ea-ab594b994f41
# ╠═14e8024c-7192-11eb-1c76-83a28b6120ee
# ╠═72a34b30-7192-11eb-3cda-15a4e9a5244c
# ╠═c264c3ec-7192-11eb-24ef-c797c42d3671
# ╠═1f6d0112-7193-11eb-20a2-efd1550ca8e2
# ╠═bd39bf8e-7193-11eb-23c0-257408e760a0
# ╠═45479604-7193-11eb-068e-7ddd263e9be2
# ╠═89ce6cb2-7193-11eb-25bb-f161f4514a6d
# ╠═d34ab594-7193-11eb-1bed-01a763445a7e
# ╟─508db812-7194-11eb-3fde-ad654ba107d1
# ╠═6732d8e0-7194-11eb-3506-93329a94dd84
# ╟─c9a0a836-7194-11eb-027a-7bfbfe6cae3c
# ╠═3deaf822-7195-11eb-2bfb-9f51b24982ed
# ╠═9dc7a434-7195-11eb-35a1-e9f79dd4ee79
# ╠═a5251374-7195-11eb-368e-5d4814c8a5e2
# ╟─83c35d98-7196-11eb-0a75-b952509c4632
# ╠═d1cc2b4a-7197-11eb-10fd-8fdadba255d2
# ╠═d0d2312a-7196-11eb-3192-2158f9b7af15
# ╠═bbe4cfb8-7199-11eb-2c60-edaa2feab075
# ╟─fd7ca6fa-7199-11eb-0691-89ea26b44013
# ╠═428fadd2-719a-11eb-3f4b-d98cb77cddbe
# ╠═0ac9bda2-719a-11eb-3036-d110c21a2fcc
# ╟─afde3d9c-719a-11eb-12cb-8397b9441a65
# ╠═fa0073b6-719a-11eb-3190-dd61c2541ae4
# ╠═5caf09e6-719b-11eb-1652-8938ad992ea3
# ╠═c56e31be-719b-11eb-022d-f796b95b6b32
# ╠═d972ae42-719b-11eb-193e-675d543025c1
# ╟─266c0a0e-719c-11eb-1692-f74b2a6f5bc4
# ╟─2dfa4aee-719c-11eb-1f89-1babfa6312d2
# ╠═48377010-719c-11eb-19a7-233bf9bb4beb
# ╟─9af9983a-719e-11eb-0a18-f5493f3df6a3
# ╠═06cdc690-719e-11eb-0819-1b5fce1e3a18
# ╟─e60ea370-719f-11eb-3d56-5dcfecf4ab3e
# ╟─90a5c372-719e-11eb-3483-6140d252b0c0
# ╠═b2d5046c-719e-11eb-07db-454247474942
# ╟─fb4038a6-719f-11eb-3e40-cd5aecfb149b
# ╠═0483452a-71a0-11eb-30c0-7d56ed83e837
