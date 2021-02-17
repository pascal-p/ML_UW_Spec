### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 03cc6704-70eb-11eb-2f5f-d727783a1f53
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
end

# ╔═╡ 6d52e124-70eb-11eb-2c56-5dcea746627e
begin
	using MLJ
	using CSV
	using DataFrames
	using PlutoUI
end

# ╔═╡ b738e190-70ef-11eb-04da-8105c8390b73
using Random

# ╔═╡ a9f436f6-70fc-11eb-394e-f7a9755bb779
using Test

# ╔═╡ 28cfd44c-70fc-11eb-0155-f984980e96f6
using Printf # Pkg.add("Printf")

# ╔═╡ 62c32226-70ea-11eb-3914-730e6d2ec0a4
md"""
## C02 - w01: Simple Linear Regression

In this notebook we will use data on house sales in King County to predict house prices using simple (one input) linear regression. we will
  - compute important summary statistics
  - Write a function to compute the Simple Linear Regression weights using the closed form solution
  - Write a function to make predictions of the output given the input feature
  - Turn the regression around to predict the input given the output
  - Compare two different models for predicting house prices
  - *Finally use a predefined MLJ solution*
"""

# ╔═╡ ec3daae4-70eb-11eb-08df-416f041130f6


# ╔═╡ fa6be52e-70eb-11eb-2050-0bd51777d5d0
sales = CSV.File("../../ML_UW_Spec/C02/w01/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ 39ca1334-70ed-11eb-2ee9-5bda94ba96f5
first(sales, 7)

# ╔═╡ 53b101ac-70f1-11eb-3844-3d5f21d9b7aa
size(sales)

# ╔═╡ d2a6fa36-70f7-11eb-3817-b36402af12f2
eltype.(eachcol(sales))

# ╔═╡ 299273f6-70f4-11eb-11c9-035af4af7032
md"""
### Split data into training and testing set
"""

# ╔═╡ 55c029d2-70f4-11eb-00ac-0b7698d7b6e2
function train_test_split(df; split=0.8, seed=42) 
	Random.seed!(seed)
	(nr, nc) = size(df)
	nrp = round(Int, nr * split)
	row_ixes = shuffle(1:nr)
	df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
	df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)
	(df_train, df_test)
end

# ╔═╡ d403506c-70f4-11eb-386f-ad0152978098
sales_train, sales_test = train_test_split(sales);

# ╔═╡ fff80614-70f3-11eb-137d-cbf3fcbaf98f
first(sales_train, 7)

# ╔═╡ 24949900-70f0-11eb-24f4-bd5c33121397
first(sales_test, 3)

# ╔═╡ 11aafb94-70f4-11eb-2438-8320f2e6cb5b
size(sales_train), size(sales_test)

# ╔═╡ 1fb75280-70f8-11eb-1cc3-29a08a026bcc
# mean and sum of a column
mean(sales_train[!, :bedrooms]), sum(sales_train[!, :bedrooms])

# ╔═╡ 799514b8-70f8-11eb-0d03-e5fe5b1c0cb4
sales_train[!, :bedrooms] .* sales_train[!, :bedrooms] 

# ╔═╡ 7c9c2c1a-70f9-11eb-3ddc-d90788056518
length(sales_train[!, :bedrooms])

# ╔═╡ cc20e6e0-70f9-11eb-2b0d-7bd7f07d4733
typeof(sales_train[!, :bedrooms])

# ╔═╡ 416a6240-70f9-11eb-1962-e77cb5215607
md"""
### Build a generic simple linear regression function
"""

# ╔═╡ 5b5efd00-70f9-11eb-36f1-75d629b148e6
function slope(X, y)::Float64
	sx, sy = sum(X), sum(y)
	n = length(X)
	num = sum(X .* y) - (1/n) * sx * sy
	den = sum(X .* X) - (1/n) * sx * sx
	num / den
end

# ╔═╡ c40f3ea0-70f9-11eb-2ff0-9b1e9c4f3084
intercept(X, y, slope) = mean(y) - slope * mean(X)

# ╔═╡ 04b3c642-70fa-11eb-3e0b-b9b86108e1af
function simple_linear_regression(X, y)::Tuple{Real, Real}
	_slope = slope(X, y)
	_intercept = intercept(X, y, _slope)
	(_intercept, _slope)
end

# ╔═╡ 4e0f51e4-70fa-11eb-2511-b9813e9c82bb
begin
	test_feature = collect(1:5) 
	test_output = 1 .+ test_feature
	
	(test_intercept, test_slope) = simple_linear_regression(test_feature, test_output)
	
	@test test_intercept ≈ 1.0
	@test test_slope ≈ 1.0
end

# ╔═╡ 64258a92-70fb-11eb-08d4-b11d5f75639f
begin
	sqft_intercept, sqft_slope = simple_linear_regression(
		sales_train[!, :sqft_living], sales_train[!, :price])

	with_terminal() do
		@printf("Intercept: %.4f\n", sqft_intercept)
		@printf("Slope:  %.4f\n", sqft_slope)
	end
end

# ╔═╡ beee3bd2-70fc-11eb-38cd-e7e8fda2f883
md"""
### Predicting Values

Now that we have the model parameters: intercept and slope we can make predictions.

Create a function to return the predicted output given the input_feature, slope and intercept.

"""

# ╔═╡ deda4470-70fc-11eb-20a2-15820ab34231
function get_regression_predictions(input_feature, intercept::Real, 
		slope::Real)
    slope .* input_feature .+ intercept
end

# ╔═╡ 1ed980ae-70fd-11eb-07ed-09b7648af658
md"""
Now that we can calculate a prediction given the slope and intercept let's make a prediction.

**Quiz Question**: Using your Slope and Intercept from above, what is the predicted price for a house with 2650 sqft?
"""

# ╔═╡ 490bda66-70fd-11eb-3cd3-e7ccd064c7ec
with_terminal() do
	house_sqft = 2650
	estimated_price = get_regression_predictions(house_sqft, 
		sqft_intercept, sqft_slope)
	
	@printf("The estimated price for a house with %d squarefeet is %.2f\n", 
		house_sqft, 
		estimated_price)
end

# ╔═╡ 96f5469a-70fd-11eb-1f17-675eeff21970
md"""
### Residual Sum of Squares

Now that we have a model and can make predictions let's evaluate our model using Residual Sum of Squares (RSS). Recall that RSS is the sum of the squares of the residuals and the residuals is just a fancy word for the difference between the predicted output and the true output.

Create a function (`get_residual_sum_of_squares`) to compute the RSS of a simple linear regression model given the input_feature, output, intercept and slope.

"""

# ╔═╡ b5e12f56-70fd-11eb-1bf7-cbdfa36fc673
function get_residual_sum_of_squares(input_feature, output, 
		intercept::Real, slope::Real)
    ## 1. get the predictions
    yhat = get_regression_predictions(input_feature, intercept, slope)
    diff = (yhat .- output)   ## 2. compute the residuals 
    rss = sum(diff .* diff)  ## 3. square the residuals and add them up
    return rss
end

# ╔═╡ 25479c96-70ff-11eb-2b86-29c052eb2f87
test_slope

# ╔═╡ 064ec37c-70fe-11eb-3bce-a79f7208c876
md"""
Let's test our `get_residual_sum_of_squares` function by applying it to the test model where the data lie exactly on a line. Since they lie exactly on a line the residual sum of squares should be zero!
"""

# ╔═╡ 1b0643d0-70fe-11eb-250c-53e54279e747
@test get_residual_sum_of_squares(test_feature, test_output, 
	test_intercept, test_slope) ≈ 0.0

# ╔═╡ 5e2a432e-70ff-11eb-3e61-bffdc0ebd46e
md"""
Now use your function to calculate the RSS on training data from the squarefeet model calculated above.

**Quiz Question**: According to this function and the slope and intercept from the squarefeet model What is the RSS for the simple linear regression using squarefeet to predict prices on *training* data?

"""

# ╔═╡ 7e8bcf82-70ff-11eb-1c79-a56db6635447
begin
	rss_prices_on_sqft = get_residual_sum_of_squares(sales_train[!, :sqft_living], 
		sales_train[!, :price], 
        sqft_intercept, 
		sqft_slope)
	with_terminal() do
		@printf("The RSS of predicting Prices based on Square Feet is: %1.3e\n", rss_prices_on_sqft)
	end
end

# ╔═╡ 0afc9a02-7100-11eb-0578-8b470fbb1d32
md"""
### Predict the squarefeet given price


What if we want to predict the squarefoot given the price? 

Since we have an equation $y = a + b \times x$ we can solve the function for x, so that if we have the intercept (a) and the slope (b) and the price (y) we can solve for the estimated squarefeet (x) with: 
$x = \frac{y - a}{b}$.

Create a function (`inverse_regression_predictions`) to compute the inverse regression estimate, i.e. predict the input_feature given the output.
"""

# ╔═╡ 5936bff2-7100-11eb-3673-03eb5174f2b0
function inverse_regression_predictions(output, intercept::Real, slope::Real)
    (output .- intercept) ./ slope
end

# ╔═╡ 7f548db8-7100-11eb-1a71-cfa1b2817f83
md"""
Now that we have a function to compute the squarefeet given the price from our simple regression model let's see how big we might expect a house that costs \$800,000 to be.

**Quiz Question**: According to this function and the regression slope and intercept from (3) what is the estimated square-feet for a house costing \$800,000?

"""

# ╔═╡ 967a4d8e-7100-11eb-369c-a7eef43335a2
begin
	house_price = 800000
	estimated_squarefeet = inverse_regression_predictions(house_price, 
		sqft_intercept, 
		sqft_slope)
	
	with_terminal() do
		@printf("The estimated squarefeet for a house worth %.2f is %.3f\n", house_price, estimated_squarefeet)
	end
end

# ╔═╡ f8cb4844-7100-11eb-2036-f1691f9277ba
md"""
### New Model: estimate prices from bedrooms

We have made one model for predicting house prices using squarefeet, but there are many other features in the sales DataFrame. 

Use your simple linear regression function to estimate the regression parameters from predicting Prices based on number of bedrooms. Use the *training* data!

"""

# ╔═╡ 1a89af3e-7101-11eb-274d-43bbcf50db13
bedr_intercept, bedr_slope = simple_linear_regression(
	sales_train[!, :bedrooms], 
	sales_train[!, :price]
)

# ╔═╡ 4d0ca242-7101-11eb-1001-7bb71d3b8de6
md"""
### Test your Linear Regression Algorithm

Now we have two models for predicting the price of a house. How do we know which one is better? Calculate the RSS on the *test* data (remember this data wasn't involved in learning the model). Compute the RSS from predicting prices using bedrooms and from predicting prices using squarefeet.

**Quiz Question**: Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about why this might be the case.

"""

# ╔═╡ 69a38e9e-7101-11eb-2e57-83829ceb7ab9
begin
	## Compute RSS when using bedrooms on test data:
	rss_prices_on_bedr = get_residual_sum_of_squares(sales_test[!, :bedrooms], 
        sales_test[!, :price], 
        bedr_intercept, 
		bedr_slope)
	with_terminal() do
		@printf("The RSS of predicting Prices based on number of Bedrooms is: %1.3e\n ", rss_prices_on_bedr)
	end
end

# ╔═╡ d043648a-7101-11eb-1b30-8d7ff3b37b7a
begin
	## Compute RSS when using squarefeet on test data:
	rss_prices_on_sqft_t = get_residual_sum_of_squares(sales_test[!, :sqft_living], 
        sales_test[!, :price], 
        sqft_intercept, 
		sqft_slope)
	with_terminal() do
		@printf("The RSS of predicting Prices based on Squarefeet is: %1.3e\n ", rss_prices_on_sqft_t)
	end
end

# ╔═╡ Cell order:
# ╟─62c32226-70ea-11eb-3914-730e6d2ec0a4
# ╠═03cc6704-70eb-11eb-2f5f-d727783a1f53
# ╠═6d52e124-70eb-11eb-2c56-5dcea746627e
# ╟─ec3daae4-70eb-11eb-08df-416f041130f6
# ╠═fa6be52e-70eb-11eb-2050-0bd51777d5d0
# ╠═39ca1334-70ed-11eb-2ee9-5bda94ba96f5
# ╠═53b101ac-70f1-11eb-3844-3d5f21d9b7aa
# ╠═d2a6fa36-70f7-11eb-3817-b36402af12f2
# ╟─299273f6-70f4-11eb-11c9-035af4af7032
# ╠═b738e190-70ef-11eb-04da-8105c8390b73
# ╠═55c029d2-70f4-11eb-00ac-0b7698d7b6e2
# ╠═d403506c-70f4-11eb-386f-ad0152978098
# ╠═fff80614-70f3-11eb-137d-cbf3fcbaf98f
# ╠═24949900-70f0-11eb-24f4-bd5c33121397
# ╠═11aafb94-70f4-11eb-2438-8320f2e6cb5b
# ╠═1fb75280-70f8-11eb-1cc3-29a08a026bcc
# ╠═799514b8-70f8-11eb-0d03-e5fe5b1c0cb4
# ╠═7c9c2c1a-70f9-11eb-3ddc-d90788056518
# ╠═cc20e6e0-70f9-11eb-2b0d-7bd7f07d4733
# ╟─416a6240-70f9-11eb-1962-e77cb5215607
# ╠═5b5efd00-70f9-11eb-36f1-75d629b148e6
# ╠═c40f3ea0-70f9-11eb-2ff0-9b1e9c4f3084
# ╠═04b3c642-70fa-11eb-3e0b-b9b86108e1af
# ╠═a9f436f6-70fc-11eb-394e-f7a9755bb779
# ╠═4e0f51e4-70fa-11eb-2511-b9813e9c82bb
# ╠═28cfd44c-70fc-11eb-0155-f984980e96f6
# ╠═64258a92-70fb-11eb-08d4-b11d5f75639f
# ╟─beee3bd2-70fc-11eb-38cd-e7e8fda2f883
# ╠═deda4470-70fc-11eb-20a2-15820ab34231
# ╟─1ed980ae-70fd-11eb-07ed-09b7648af658
# ╠═490bda66-70fd-11eb-3cd3-e7ccd064c7ec
# ╟─96f5469a-70fd-11eb-1f17-675eeff21970
# ╠═b5e12f56-70fd-11eb-1bf7-cbdfa36fc673
# ╠═25479c96-70ff-11eb-2b86-29c052eb2f87
# ╟─064ec37c-70fe-11eb-3bce-a79f7208c876
# ╠═1b0643d0-70fe-11eb-250c-53e54279e747
# ╟─5e2a432e-70ff-11eb-3e61-bffdc0ebd46e
# ╠═7e8bcf82-70ff-11eb-1c79-a56db6635447
# ╟─0afc9a02-7100-11eb-0578-8b470fbb1d32
# ╠═5936bff2-7100-11eb-3673-03eb5174f2b0
# ╟─7f548db8-7100-11eb-1a71-cfa1b2817f83
# ╠═967a4d8e-7100-11eb-369c-a7eef43335a2
# ╟─f8cb4844-7100-11eb-2036-f1691f9277ba
# ╠═1a89af3e-7101-11eb-274d-43bbcf50db13
# ╟─4d0ca242-7101-11eb-1001-7bb71d3b8de6
# ╠═69a38e9e-7101-11eb-2e57-83829ceb7ab9
# ╠═d043648a-7101-11eb-1b30-8d7ff3b37b7a
