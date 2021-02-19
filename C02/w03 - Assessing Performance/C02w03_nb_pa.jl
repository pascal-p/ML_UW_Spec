### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 5dcfce72-7255-11eb-0aba-778dd1e4d053
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
end

# ╔═╡ 6f773f48-7255-11eb-1d7d-ebad5f59c12c
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

# ╔═╡ 2fd9b8c0-7255-11eb-1648-07c867431b74
md"""
# Week3: Assessing fit (Polynomial Regression)

*Author: Pascal,  Feb 2021*

In this notebook you will compare different regression models in order to assess which model fits best. We will be using polynomial regression as a mean to examine this topic. In particular we will:
  - Write a function to take a Vector and a degree and return an DataFrame where each column is the Vector to a polynomial value up to the total degree e.g. degree = 3 then column 1 is the Vector column 2 is the Vector squared and column 3 is the Vector cubed
  - *Use Plots to visualize polynomial regressions*
  - *Use Plots to visualize the same polynomial degree on different subsets of the data*
  - Use a validation set to select a polynomial degree
  - Assess the final fit using test data

We will continue to use the House data from previous notebooks.
"""

# ╔═╡ 2c111f6e-7258-11eb-1ce3-1582510c7875
md"""
### Polynomial dataframe function
"""

# ╔═╡ f242f3c2-7255-11eb-1bd7-53b9ebc336df
function polynomial_df(feature; degree=3)
	@assert degree ≥ 1 "Expect degree to be ≥ 1"
	
	hsh = Dict{Symbol, Vector{Float64}}(:power_1 => feature)
	for deg ∈ 2:degree
		hsh[Symbol("power_$(deg)")] = feature .^ deg	
	end
	
	return DataFrame(hsh)
end

# ╔═╡ a6827a5a-7257-11eb-32ee-3d43fcf750a0
v = Vector{Float64}([10., 4., 9.])

# ╔═╡ cb48775e-7257-11eb-28b3-93708cf4f39f
df = polynomial_df(v; degree=4)

# ╔═╡ 42b14492-7258-11eb-1b78-99bbac50f8bc
md"""
### Visualizing polynomial regression
"""

# ╔═╡ 4ef1490a-7258-11eb-0b6b-0fb70d911779
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ ac9cd006-7258-11eb-36da-87759d7d59ef
first(sales, 5)

# ╔═╡ 67424342-7258-11eb-186a-01cd28c1a0a7
function train_test_split(df; split=0.8, seed=42) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffle(1:nr)
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	(df_train, df_test)
end

# ╔═╡ 7402efd2-7258-11eb-283f-95834d032f9a
begin
	sort!(sales, [:sqft_living, :price], rev=[false, false]);
	first(sales, 5)
end

# ╔═╡ ea3e6bea-7258-11eb-294b-fbca040fbd73
md"""
##### First degree polynomial

Let's start with a degree 1 polynomial, using `:sqft_living` to predict `:price` nad plot what it looks like.
"""

# ╔═╡ 1a2e016e-7259-11eb-11f7-ebda74351974
begin
	poly_df_1 = polynomial_df(sales.sqft_living; degree=1)
	poly_df_1[!, :price] = sales.price
	
	first(poly_df_1, 3)
end

# ╔═╡ d34d46fa-725a-11eb-2530-b5efd1bc0c4a
@load LinearRegressor pkg=MLJLinearModels

# ╔═╡ 37bd7248-725a-11eb-2d75-15427b89b31c
begin
	mdl1 = MLJLinearModels.LinearRegressor()
	
	X_1 = select(poly_df_1, :power_1)
	y_1 = poly_df_1.price
	
	mach1 = machine(mdl1, X_1, y_1)
	fit!(mach1)
	fp1 = fitted_params(mach1)
	
	with_terminal() do
		for (name, c) in fp1.coefs
   			println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
		end
		
		println("Intercept: $(round(fp1.intercept, sigdigits=3))")
	end
end

# ╔═╡ 94e85b12-7260-11eb-0764-a13313833a2b
typeof(X_1), typeof(y_1)

# ╔═╡ 992e1d44-7259-11eb-36c2-67821627b331
begin
  	scatter(X_1.power_1, y_1, marker=".")
 	plot!(X_1.power_1, predict(mach1, X_1), marker="-")
end

# ╔═╡ 4b509104-7263-11eb-01a2-27074174de99
md"""
We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 275 and intercept -28600.


##### Second degree polynomial
"""

# ╔═╡ 95fd53b0-7263-11eb-317d-cf09779af7de
begin
	poly_df_2 = polynomial_df(sales.sqft_living; degree=2)
	feature_df_2 = names(poly_df_2)
	poly_df_2[!, :price] = sales.price
	
	first(poly_df_2, 3)
end

# ╔═╡ 774dba36-7263-11eb-280b-59e3c0967350
begin
	mdl2 = MLJLinearModels.LinearRegressor()
	
	X_2 = select(poly_df_2, feature_df_2)
	y_2 = poly_df_2.price
	
	mach2 = machine(mdl2, X_2, y_2)
	fit!(mach2)
	fp2 = fitted_params(mach2)
	
	with_terminal() do
		for (name, c) in fp2.coefs
   			println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
		end
		
		println("Intercept: $(round(fp2.intercept, sigdigits=3))")
	end
end

# ╔═╡ 6bf2644c-7264-11eb-3cbb-63d558bce5e7
## Visualization
begin
	scatter(X_2.power_1, y_2, legend=false, color=[:lightblue], marker=".")
 	plot!(X_2.power_1, predict(mach2, X_2), marker="-")
end

# ╔═╡ 9ab047c6-7265-11eb-28bd-e5597084eb01
md"""
##### Third degree polynomial
"""

# ╔═╡ b1dc1cc2-7265-11eb-29d0-67171da70faf
begin
	poly_df_3 = polynomial_df(sales.sqft_living; degree=3)
	feature_df_3 = names(poly_df_3)
	poly_df_3[!, :price] = sales.price
	
	mdl3 = MLJLinearModels.LinearRegressor()
	
	X_3 = select(poly_df_3, feature_df_3)
	y_3 = poly_df_3.price
	
	mach3 = machine(mdl3, X_3, y_3)
	fit!(mach3)
	fp3 = fitted_params(mach3)
	
	with_terminal() do
		for (name, c) in fp3.coefs
   			println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
		end
		
		println("Intercept: $(round(fp3.intercept, sigdigits=3))")
	end
end

# ╔═╡ 06defbd6-7266-11eb-22f7-c3f26eed97d1
## Visualization
begin
	scatter(X_3.power_1, y_3, legend=false, color=[:lightblue], marker=".")
 	plot!(X_3.power_1, predict(mach3, X_3), marker="-")
end

# ╔═╡ 30e47bb6-7266-11eb-381b-692ccdaa28e0
md"""
##### 15th degree polynomial
"""

# ╔═╡ 53617e0a-7266-11eb-11da-1db6f06abb27
begin
	poly_df_15 = polynomial_df(sales.sqft_living; degree=15)
	feature_df_15 = names(poly_df_15)
	poly_df_15[!, :price] = sales.price
	
	mdl15 = MLJLinearModels.LinearRegressor()
	
	X_15 = select(poly_df_15, feature_df_15)
	y_15 = poly_df_15.price
	
	mach15 = machine(mdl15, X_15, y_15)
	fit!(mach15)
	fp15 = fitted_params(mach15)
	
	with_terminal() do
		for (name, c) in fp15.coefs
   			println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
		end
		
		println("Intercept: $(round(fp15.intercept, sigdigits=3))")
	end
end

# ╔═╡ 9be4a94c-7266-11eb-07ef-4526f53c5d1a
## Visualization
begin
	scatter(X_15.power_1, y_15, legend=false, color=[:lightblue], marker=".")
 	plot!(X_15.power_1, predict(mach15, X_15), marker="-")
end

# ╔═╡ e4e3f5a8-7266-11eb-107c-0734e0631801
md"""
What do you think of the 15th degree polynomial? <br />
Do you think this is appropriate? <br />

*As expected, it looks like the model learn too much of the idiosyncrasies of the training data*


If we were to change the data do you think you'd get pretty much the same curve? Let's take a look.
"""

# ╔═╡ 03b2adb2-7267-11eb-0f0d-5fe0b5a92d83
md"""
## Changing the data and re-learning

We are going to split the sales data into four subsets of roughly equal size. Then you will estimate a 15th degree polynomial model on all four subsets of the data. Print the coefficients and plot the resulting fit (as we did above). The quiz will ask you some questions about these results.

To split the sales data into four subsets, we perform the following steps:
* First split sales into 2 subsets, 50% of the original set
* Next split the resulting subsets into 2 more subsets each.

We set `seed=42` in these steps so that different users get consistent results.
we should end up with 4 subsets (`set1`, `set2`, `set3`, `set4`) of approximately equal size. 
"""

# ╔═╡ d10379cc-7267-11eb-1d18-2fd3c85988b7
begin
	(ssales_a, ssales_b) = train_test_split(sales; split=0.5, seed=42) 

	(set1, set2) = train_test_split(ssales_a; split=0.5, seed=42)
	(set3, set4) = train_test_split(ssales_b; split=0.5, seed=42)

	(size(set1), size(set2), size(set3), size(set4))
end

# ╔═╡ a012e338-7268-11eb-3fb8-b9f5183c4c68
md"""
Fit a 15th degree polynomial on `set1`, `set2`, `set3`, and `set4` using `sqft_living` to predict prices.

Print the coefficients and make a plot of the resulting model.
"""

# ╔═╡ f8554ba8-7268-11eb-1982-21b027f2ebdc
begin

function make_poly(tset, degree)
  poly_df = polynomial_df(tset.sqft_living; degree) 
  features = names(poly_df)   
  poly_df[!, :price] = tset.price
  (features, poly_df)
end

function fit_poly(tset; degree=15)
  (features, poly_df) = make_poly(tset, degree)  
  mdl = MLJLinearModels.LinearRegressor()	
  X_ = select(poly_df, features)
  y_ = poly_df.price
	
  mach = machine(mdl, X_, y_)
  fit!(mach)
 
  (mach, X_, y_)
end
	
function print_coeff(mach)
  fp = fitted_params(mach)
	
  with_terminal() do
	for (name, c) in fp.coefs
	  println("$(rpad(name, 10)):  $(round(c, sigdigits=3))")
    end
		
	println("Intercept: $(round(fp.intercept, sigdigits=3))")
  end
end
	
end

# ╔═╡ 3894cde6-726a-11eb-2839-8f76807bc97c
md"""
##### Model for set1
"""

# ╔═╡ 572f9ac4-726a-11eb-0697-41315716e42c
## set 1
begin
	(mach_set1, Xset1, yset1) = fit_poly(set1)
	print_coeff(mach_set1)
end

# ╔═╡ 572e63ac-726a-11eb-1224-b76bbb638a81
## Visualization
begin
	scatter(Xset1.power_1, yset1, legend=false, color=[:lightblue], marker=".")
 	plot!(Xset1.power_1, predict(mach_set1, Xset1), marker="-")
end

# ╔═╡ 572e35b2-726a-11eb-3a30-094923a3c336
md"""
##### Model for set2
"""

# ╔═╡ 52786aee-726a-11eb-3b05-612363f3dbd3
## set 2
begin
	(mach_set2, Xset2, yset2) = fit_poly(set2)
	print_coeff(mach_set2)
end

# ╔═╡ 51f1196e-726a-11eb-028d-f71e46dd9101
## Visualization
begin
	scatter(Xset2.power_1, yset2, legend=false, color=[:lightblue], marker=".")
 	plot!(Xset2.power_1, predict(mach_set2, Xset2), marker="-")
end

# ╔═╡ 847ce18a-726a-11eb-22cb-a3c677741317
md"""
##### Model for set3
"""

# ╔═╡ 8064e2c0-726a-11eb-0928-f90b0e4c1193
## set 3
begin
	(mach_set3, Xset3, yset3) = fit_poly(set3)
	print_coeff(mach_se3)
end

# ╔═╡ 800e299c-726a-11eb-0181-0507645ce3be
## Visualization
begin
	scatter(Xset3.power_1, yset3, legend=false, color=[:lightblue], marker=".")
 	plot!(Xset3.power_1, predict(mach_set3, Xset3), marker="-")
end

# ╔═╡ 7fb58b3e-726a-11eb-27db-87d8b8134a3f
md"""
##### Model for set4
"""

# ╔═╡ 7e5df8fc-726a-11eb-16c2-ad1be0706721
## set 4
begin
	(mach_set4, Xset4, yset4) = fit_poly(set4)
	print_coeff(mach_set4)
end

# ╔═╡ 6b6de5e4-726b-11eb-3b84-170b5bca19d8
## Visualization
begin
	scatter(Xset4.power_1, yset4, legend=false, color=[:lightblue], marker=".")
 	plot!(Xset4.power_1, predict(mach_set4, Xset4), marker="-")
end

# ╔═╡ d1854742-726c-11eb-0bb1-7dfb5af6af6e
md"""
Some questions you will be asked on your quiz:

**Quiz Question: Is the sign (positive or negative) for power_15 the same in all four models?**
  - Sign is positive for model 1, 2 and 3 and negative for model 4 
 
**Quiz Question: (True/False) the plotted fitted lines look the same in all four plots**
  - The fitted lines are very different for each model.
"""

# ╔═╡ afb4130c-726d-11eb-06fb-c75c713c5f5e
md"""
## Selecting a Polynomial Degree
Whenever we have a "magic" parameter like the degree of the polynomial there is one well-known way to select these parameters: validation set. (We will explore another approach in week 4).

We split the sales dataset 3-way into training set, test set, and validation set as follows:

* Split our sales data into 2 sets: `training_and_validation` and `testing`. Use 90%/10% split.
* Further split our training data into two sets: `training` and `validation`. Use 50%/50% split.

We set `seed=42` to obtain consistent results for different users.

"""

# ╔═╡ 350f14ac-726e-11eb-18ce-9379b4e8b243
begin
	(training_validation, testing) = train_test_split(sales; split=0.9, seed=42) 
	(training, validation) = train_test_split(training_validation; split=0.6, seed=42)

	(size(training), size(validation), size(testing))
end

# ╔═╡ 33d6443c-726e-11eb-2d4b-9561d4f67631
md"""
Next you should write a loop that does the following:
* For degree ∈ 1:15 
    * Build a DataFrame of polynomial data of `train_data.sqft_living` at the current degree
    
    * Add `train_data.price` to the polynomial DataFrame
    * Learn a polynomial regression model to sqft vs price with that degree on *training* data
    * Compute the RSS on *validation* data for that degree and you will need to make a polynmial DataFrame using *validation* data.
* Report which degree had the lowest RSS on validation data 
"""

# ╔═╡ 32dfe6a2-726e-11eb-2218-3df42e232960
function get_rss(mach, X, y)
    ŷ = predict(mach, X)     # First get the predictions
    diff = y .- ŷ            # Then compute the residuals/errors
    rss = sum(diff .* diff)  # Then square and add them up
    return rss
end

# ╔═╡ 31e97fc4-726e-11eb-2d37-4dc1f85c598f
function find_best_degree()
	max_degree = 15
	best_rss = nothing
	best_degree, best_mach = (nothing, nothing)
	
	for degree ∈ 1:max_degree
  		(mach_set, _Xset, _yset) = fit_poly(training; degree)
  		(_features_val, poly_df_val) = make_poly(validation, degree)
  		rss = get_rss(mach_set, poly_df_val, poly_df_val.price)	
  		# @printf("degree: %2d / rss: %2.5e / best rss: %2.5e\n", degree, rss, best_rss)	
  		if isnothing(best_rss) || rss < best_rss
    		best_rss = rss
    		best_degree = degree
    		best_mach = mach_set
		end
	end
	return (best_degree, best_rss, best_mach)
end

# ╔═╡ 2b4109de-7274-11eb-0060-017c0b721a65
begin
	(best_degree, best_rss, best_mach) = find_best_degree()
	with_terminal() do
		@printf("best model degree: %2d / lowest rss: %2.5e\n", best_degree, best_rss)
	end
end

# ╔═╡ 14527cd2-7272-11eb-2d8c-65a9b24e298e
md"""

**Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on Validation data?**

Now that you have chosen the degree of your polynomial using validation data, compute the RSS of this model on *testing* data. Report the RSS on your quiz.

"""

# ╔═╡ 56781eb4-7272-11eb-0460-113b61c44237
with_terminal() do
	(_features_test, poly_df_test) = make_poly(testing, best_degree)
 	rss_test = get_rss(best_mach, poly_df_test, poly_df_test.price)
	
	@printf("best model degree: %2d / rss: on test set %2.5e\n", best_degree, rss_test)
end

# ╔═╡ Cell order:
# ╠═2fd9b8c0-7255-11eb-1648-07c867431b74
# ╠═5dcfce72-7255-11eb-0aba-778dd1e4d053
# ╠═6f773f48-7255-11eb-1d7d-ebad5f59c12c
# ╟─2c111f6e-7258-11eb-1ce3-1582510c7875
# ╠═f242f3c2-7255-11eb-1bd7-53b9ebc336df
# ╠═a6827a5a-7257-11eb-32ee-3d43fcf750a0
# ╠═cb48775e-7257-11eb-28b3-93708cf4f39f
# ╟─42b14492-7258-11eb-1b78-99bbac50f8bc
# ╠═4ef1490a-7258-11eb-0b6b-0fb70d911779
# ╠═ac9cd006-7258-11eb-36da-87759d7d59ef
# ╠═67424342-7258-11eb-186a-01cd28c1a0a7
# ╠═7402efd2-7258-11eb-283f-95834d032f9a
# ╠═ea3e6bea-7258-11eb-294b-fbca040fbd73
# ╠═1a2e016e-7259-11eb-11f7-ebda74351974
# ╠═d34d46fa-725a-11eb-2530-b5efd1bc0c4a
# ╠═37bd7248-725a-11eb-2d75-15427b89b31c
# ╠═94e85b12-7260-11eb-0764-a13313833a2b
# ╠═992e1d44-7259-11eb-36c2-67821627b331
# ╟─4b509104-7263-11eb-01a2-27074174de99
# ╠═95fd53b0-7263-11eb-317d-cf09779af7de
# ╠═774dba36-7263-11eb-280b-59e3c0967350
# ╠═6bf2644c-7264-11eb-3cbb-63d558bce5e7
# ╟─9ab047c6-7265-11eb-28bd-e5597084eb01
# ╠═b1dc1cc2-7265-11eb-29d0-67171da70faf
# ╠═06defbd6-7266-11eb-22f7-c3f26eed97d1
# ╟─30e47bb6-7266-11eb-381b-692ccdaa28e0
# ╠═53617e0a-7266-11eb-11da-1db6f06abb27
# ╠═9be4a94c-7266-11eb-07ef-4526f53c5d1a
# ╟─e4e3f5a8-7266-11eb-107c-0734e0631801
# ╟─03b2adb2-7267-11eb-0f0d-5fe0b5a92d83
# ╠═d10379cc-7267-11eb-1d18-2fd3c85988b7
# ╟─a012e338-7268-11eb-3fb8-b9f5183c4c68
# ╠═f8554ba8-7268-11eb-1982-21b027f2ebdc
# ╟─3894cde6-726a-11eb-2839-8f76807bc97c
# ╠═572f9ac4-726a-11eb-0697-41315716e42c
# ╠═572e63ac-726a-11eb-1224-b76bbb638a81
# ╟─572e35b2-726a-11eb-3a30-094923a3c336
# ╠═52786aee-726a-11eb-3b05-612363f3dbd3
# ╠═51f1196e-726a-11eb-028d-f71e46dd9101
# ╟─847ce18a-726a-11eb-22cb-a3c677741317
# ╠═8064e2c0-726a-11eb-0928-f90b0e4c1193
# ╠═800e299c-726a-11eb-0181-0507645ce3be
# ╟─7fb58b3e-726a-11eb-27db-87d8b8134a3f
# ╠═7e5df8fc-726a-11eb-16c2-ad1be0706721
# ╠═6b6de5e4-726b-11eb-3b84-170b5bca19d8
# ╟─d1854742-726c-11eb-0bb1-7dfb5af6af6e
# ╟─afb4130c-726d-11eb-06fb-c75c713c5f5e
# ╠═350f14ac-726e-11eb-18ce-9379b4e8b243
# ╟─33d6443c-726e-11eb-2d4b-9561d4f67631
# ╠═32dfe6a2-726e-11eb-2218-3df42e232960
# ╠═31e97fc4-726e-11eb-2d37-4dc1f85c598f
# ╠═2b4109de-7274-11eb-0060-017c0b721a65
# ╟─14527cd2-7272-11eb-2d8c-65a9b24e298e
# ╠═56781eb4-7272-11eb-0460-113b61c44237
