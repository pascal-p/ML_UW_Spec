### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ a450c12c-753d-11eb-06fc-29c5d907cb06
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
	
	# using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	using Random
	using Test
	using Printf
	# using Plots
end

# ╔═╡ a4ce9686-7549-11eb-2172-2d47391d1260
include("./utils.jl")

# ╔═╡ 5b0c59ae-753d-11eb-1a8a-6fa0bf99a357
md"""
## Week 5: Lasso (Coordinate Descent)

In this notebook, you will implement your very own LASSO solver via coordinate descent. We will:

  - Write a function to normalize features
  - Implement coordinate descent for Lasso
  - Explore effects of L1 penalty


"""

# ╔═╡ 7e12df34-7549-11eb-30a5-b3099a8f77f6
md"""
### Load in house sales data
"""

# ╔═╡ 9e5e1060-7549-11eb-0418-55bcd0b8c757
sales = CSV.File("../../ML_UW_Spec/C02/data/kc_house_test_data.csv"; 
	header=true) |> DataFrame;

# ╔═╡ 2fa16676-754a-11eb-00cc-896d0e28b246
md"""
### Normalize features

In the house dataset, features vary wildly in their relative magnitude: sqft_living is very large overall compared to bedrooms, for instance. As a result, weight for sqft_living would be much smaller than weight for bedrooms. This is problematic because "small" weights are dropped first as l1_penalty goes up.

To give equal considerations for all features, we need to normalize features as discussed in the lectures: we divide each feature by its 2-norm so that the transformed feature has norm 1.
"""

# ╔═╡ 4e5d02be-754a-11eb-061f-41ab2f747c54
md"""
moved to `utils.jl`

```julia
using LinearAlgebra
	
function normalize_features(X::Matrix)
  n = size(X)[2]
  norms = zeros(eltype(X), n)'
  for ix ∈ 1:n
    norms[ix] = norm(X[:, ix])
  end
  (X ./ norms, norms)
end
```
"""

# ╔═╡ dc99317e-754a-11eb-0a8e-61ec5dff0507
begin
	M1 = [3. 6. 9.; 4. 8. 12.]
	features1, norms1 = normalize_features(M1)
	
	@test features1 == [0.6 0.6 0.6; 0.8 0.8 0.8]
	@test norms1 == [5.  10.  15.]  ## or [5.,  10.,  15.]
end

# ╔═╡ e9128af8-7555-11eb-1f4d-d1e17d70f4b3
typeof(features1), typeof(norms1)

# ╔═╡ 0d7f1fa0-754c-11eb-330b-d31a86f5566c
begin
	# 3 rows x 2 cols
	M2 = [3. 4.; 6. 5.; 9. 10.]
	features2, norms2 = normalize_features(M2)
	
	@test features2 ≈ [0.2672612419124244 0.3368607684266076; 
		0.5345224838248488 0.42107596053325946; 
		0.8017837257372732 0.8421519210665189]
	@test norms2 ≈ [11.224972160321824 11.874342087037917]
end

# ╔═╡ 984c778a-7557-11eb-21f8-dd2502fe7b73
md"""
### Implementing Coordinate Descent with normalized features

We seek to obtain a sparse set of weights by minimizing the LASSO cost function

$$\Sigma (prediction - output)^2 + \lambda \times \Sigma_{j=1}^{k} | w_j |$$ 

(By convention, we do not include $w_0$ in the L1 penalty term. We never want to push the intercept to zero.)

The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable (you would need to implement a method called subgradient descent). Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but weight `i` and find the value of weight `i` that minimizes the objective. That is, we look for

$$argmin_{w_i}[\Sigma (prediction - output)^2 + \lambda \times \Sigma_{j=1}^{k} | w_j |]$$

where all weights other than $w_i$ are held to be constant. We will optimize one $w_i$ at a time, circling through the weights multiple times.  
  1. Pick a coordinate `i`
  2. Compute $w_i$ that minimizes the cost function $\Sigma[ (prediction - output)^2 ] + \lambda \times \Sigma_{j=1}^{k} | w_j |$
  3. Repeat Steps 1 and 2 for all coordinates, multiple times


For this notebook, we use **cyclical coordinate descent with normalized features**, where we cycle through coordinates 0 to (d-1) in order, and assume the features were normalized as discussed above. The formula for optimizing each coordinate is as follows:

$$w_i = \left\{
    \begin{array}{lr}
      \rho_i + \frac{\lambda}{2} & if \rho_i < -\frac{\lambda}{2} \\
       0  &                        if -\frac{\lambda}{2} \le \rho_i \le \frac{\lambda}{2} \\
      \rho_i - \frac{\lambda}{2} & if \rho_i > \frac{\lambda}{2} \\
    \end{array}
\right.$$

where $\rho_i = \Sigma [ [feature_i] \times (output - prediction + w\_i \times [feature_i]) ]$

Note that we do not regularize the weight of the constant feature (intercept) `w[0]`, so, for this weight, the update is simply:
$w_0 = \rho_i$

"""

# ╔═╡ eb96bfc8-7556-11eb-2045-0f0e7adabe76
md"""
#### Effect of L1 penalty
"""

# ╔═╡ 9c8d9610-754f-11eb-05c3-edd271661b84
begin
	simple_features = [:sqft_living, :bedrooms]
	output_ = :price
	(simple_feature_matrix, output) = get_data(sales, simple_features, output_);
end

# ╔═╡ eb872178-754f-11eb-177d-7bb5ee1046f5
begin
	simple_feature_matrix_n, norms_n = normalize_features(simple_feature_matrix);
end

# ╔═╡ 20799d0c-7550-11eb-16ee-676be88a80af
typeof(simple_feature_matrix), typeof(simple_feature_matrix_n)

# ╔═╡ 06be0f04-7557-11eb-3f7c-bfc5e5a172d4
md"""
We assign some random set of initial weights and inspect the values of ρ[i].

we use `predict_output()` to make predictions on this data.
"""

# ╔═╡ 06a0f408-7557-11eb-182e-8312809145d1
begin
	weights₀ = [1., 4., 1.]

	prediction = predict_output(simple_feature_matrix_n, weights₀);
	prediction, size(prediction)
end

# ╔═╡ 06814074-7557-11eb-2359-8fa03900e3f8
md"""
Compute the values of `ρ[i]` for each feature in this simple model, using the formula given above, using the formula:

$\rho_i = \Sigma [ [feature_i] \times (output - prediction + w\_i \times [feature_i]) ]$

"""

# ╔═╡ fcc38dfc-7557-11eb-048c-93e95ce9bbbc
simple_feature_matrix_n[:, 2]

# ╔═╡ 04b4ddf8-7557-11eb-0d69-f7a89e8936f8
ρ = [
	sum(simple_feature_matrix_n[:, i] .* 
		(output - prediction + weights₀[i] * simple_feature_matrix[:, i]))
	for i in 1:length(weights₀)
]

# ╔═╡ 0498db3c-7557-11eb-1af6-9de231a4256f
md"""
**Quiz Question**

Recall that, whenever `ρ[i]` falls between `-l1_penalty/2` and `l1_penalty/2`, the corresponding weight `w[i]` is sent to zero. 

Now suppose we were to take one step of coordinate descent on either feature 1 or feature 2. 

What range of values of `l1_penalty` **would not** set `w[1]` to zero, but **would** set `w[2]` to zero, if we were to take a step in that coordinate? 

"""

# ╔═╡ 0480646e-7557-11eb-3aa9-af64a4502ff9
md"""
we know that by definition:

by definition: 
  - for w[2]: $|\rho_2| \le \frac{l_1}{2} \equiv  l_1 \ge 2 * \rho_2$ 
  - for w[1]: $|\rho_1| \le \frac{l_1}{2} \equiv  l_1 \ge 2 * \rho_1$ 

"""

# ╔═╡ 046a8ad4-7557-11eb-2aa4-e37c7f37f711
## l1 range that would set w[2] to zero (but not w[1])
min(ρ[2] * 2, ρ[1] * 2)

# ╔═╡ 044fc410-7557-11eb-3a80-eb215ebc3f5f
md"""
**Quiz Question**

What range of values of l1_penalty would set both w[1] and w[2] to zero, if we were to take a step in that coordinate? 
"""

# ╔═╡ 0434c2be-7557-11eb-0592-1b733cae76ca
## Any value strictly greater than
max(ρ[2] * 2, ρ[1] * 2)

# ╔═╡ 041c6f52-7557-11eb-050e-c97dee4f3220
md"""
So we can say that `ρ[i]` quantifies the significance of the i-th feature: the larger `ρ[i]` is, the more likely it is for the i-th feature to be retained.
"""

# ╔═╡ 4f9ee984-755a-11eb-06cb-8da12ef68907
md"""
### Single Coordinate Descent Step

Using the formula above, implement coordinate descent that minimizes the cost function over a single feature i. Note that the intercept (weight 0) is not regularized. The function should accept feature matrix, output, current weights, l1 penalty, and index of feature to optimize over. The function should return new weight for feature i.
"""

# ╔═╡ 53782228-755a-11eb-1554-394795823745
function lasso_coordinate_descent_step(ix, f_matrix, output, weights, l₁)
	pred = predict_output(f_matrix, weights) 
	
	ρ_ix = sum(f_matrix[:, ix] .* (output - pred + weights[ix] * f_matrix[:, ix]))
	
	new_weight_ix = if ix == 1   ## intercept -- do not regularize
		 ρ_ix
	elseif ρ_ix < -l₁ / 2.
		ρ_ix + l₁ / 2.
	elseif ρ_ix >  l₁ / 2.
		ρ_ix - l₁ / 2.
	else
		0.
	end
	
	new_weight_ix
end

# ╔═╡ 535bf7b0-755a-11eb-06ea-8b0e0baec03c
begin
	ϵ = 1e-7
	exp = -1.7706583789422325
	act = lasso_coordinate_descent_step(
		1, 
		[3. / √(13)  1. / √(10); 2. / √(13)  3. / √(10)], 
		[1., 1.], 
		[1., 4.], 
	0.1)

	@test abs(act - exp) ≤ ϵ
end

# ╔═╡ 0403eec8-7557-11eb-18f7-13b0403e0a5b
md"""
#### Cyclical coordinate descent

Now that we have a function that optimizes the cost function over a single coordinate, let us implement cyclical coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat.

When do we know to stop? Each time we scan all the coordinates (features) once, we measure the change in weight for each coordinate. If no coordinate changes by more than a specified threshold, we stop.

For each iteration:
1. As you loop over features in order and perform coordinate descent, measure how much each coordinate changes.
2. After the loop, if the maximum change across all coordinates falls below the tolerance, stop. Otherwise, go back to step 1.

Return weights

*IMPORTANT: when computing a new weight for coordinate i, make sure to incorporate the new weights for coordinates 0, 1, ..., i-1. One good way is to update your weights variable in-place.*

"""

# ╔═╡ 03ec653c-7557-11eb-2240-8b060cbb17af
function lasso_cyclical_coordinate_descent(f_matrix, output, init_weights,
		l1_penalty, tolerance)

	weights = copy(init_weights)
	while true
		coord = []
		
		for ix in 1:length(weights)
			prev_weight_ix = weights[ix]
			weights[ix] = lasso_coordinate_descent_step(ix, f_matrix, output, 
				weights, l1_penalty)
			push!(coord, abs(prev_weight_ix - weights[ix]))
		end
		
		maximum(coord) < tolerance && break
	end
	weights
end

# ╔═╡ dcf8bdc2-755b-11eb-04da-39f89dd5709c
begin
	## utility functions, for the coming questions...

	function features_nzw(features, weights)
  		"""features with non-zero weigths"""
  		return filter(
			((_n, w)=t) -> w > 0., 
			collect(zip([:intercept, features...], weights))
		)
		
	end
	
	function features_zw(features, weights, ϵ=1e-7)
  		"""features with zero weigths"""
  		return filter(
			((_n, w)=t) -> w ≤ ϵ, 
			collect(zip([:intercept, features...], weights)) 
			# intercept is constant in the quiz
		)  
	end
end

# ╔═╡ 03d065a8-7557-11eb-1cf0-add64e6d7596
begin
	## Using the following parameters, learn the weights on the sales dataset. 
	s_features = [:sqft_living, :bedrooms]
	s_outputn = :price
	s_init_weights = zeros(Float64, length(s_features) + 1)  # +1 for intercept
	s_l1_penalty = 1e7
	s_tolerance = 1.0
	
	s_f_matrix, s_output = get_data(sales, s_features, s_outputn)
	(norm_s_f_matrix, s_norms) = normalize_features(s_f_matrix) 
	
	s_weights = lasso_cyclical_coordinate_descent(norm_s_f_matrix, s_output, 		
		s_init_weights, s_l1_penalty, s_tolerance)

	s_weights, size(s_weights)
end

# ╔═╡ cf49001e-755e-11eb-33d9-9182f04696f8
md"""
**Quiz Questions**

  1. What is the RSS of the learned model on the normalized dataset? (Hint: use the normalized feature matrix when you make predictions.)
  1. Which features had weight zero at convergence?
"""

# ╔═╡ d35515f6-755e-11eb-3da0-0bc64c7ed3b1
with_terminal() do
	## 1. RSS
	rss = calc_rss(norm_s_f_matrix, s_output, s_weights)
	@printf("rss: %.3f / in scientifc notation: %1.5e\n", rss, rss)
end

# ╔═╡ acd4c536-755e-11eb-1b0c-9573130238dc
## 2. features with weight zero
features_zw(s_features, s_weights)

# ╔═╡ a941a03a-755f-11eb-38f7-455aaa090d7f
md"""
### Evaluating LASSO fit with more features 
"""

# ╔═╡ c26ef2fe-755f-11eb-28e4-ef66e958b894
begin
	train_data, test_data = train_test_split(sales; split=0.8, seed=42) 
	size(train_data), size(test_data)
end

# ╔═╡ c251d9bc-755f-11eb-2fe2-13c496a0abbd
begin
	a_features = [
		:bedrooms, :bathrooms, :sqft_living, :sqft_lot, :floors, 
		:waterfront, :view, :condition, :grade, :sqft_above, :sqft_basement,
		:yr_built, :yr_renovated
	]
	
	tr_outputn = :price
	(tr_f_matrix, tr_output) = get_data(train_data, a_features, tr_outputn)
	(norm_tr_f_matrix, norms_tr) = normalize_features(tr_f_matrix)

	size(norm_tr_f_matrix), size(norms_tr)
end

# ╔═╡ c0ff7904-755f-11eb-2ad7-572acb587b6f
md"""
##### Learn with l₁ of 1e7
"""

# ╔═╡ c0e5f506-755f-11eb-1ad1-6906e5ccfd85
begin
	l1₁=1e7
	init_weights = zeros(length(a_features) + 1)
	tol = 1.0

	weights1e7 = lasso_cyclical_coordinate_descent(norm_tr_f_matrix, tr_output,
		init_weights, l1₁, tol)
	(size(weights1e7), typeof(weights1e7))
end

# ╔═╡ c0ca4482-755f-11eb-156b-a7011072410a
md"""
**Quiz Question**

What features had non-zero weight in this case?

```julia
features_nzw(a_features, weights1e7)
```
"""

# ╔═╡ 230c74ce-7562-11eb-12d7-43e526f2ec26
with_terminal() do
	for t ∈ features_nzw(a_features, weights1e7)
		@printf("%-20s => %2.5e / %15.5f\n", string(t[1]), t[2], t[2])
	end
end

# ╔═╡ c0b05b74-755f-11eb-319e-9f5eff250eeb
md"""
##### Learn with l₁ of 1e8
"""

# ╔═╡ c095ab50-755f-11eb-2347-79fca41a8555
begin
	l1₂=1e8
	# init_weights2 = zeros(length(a_features) + 1)
	# tol₂ = 1.0

	weights1e8 = lasso_cyclical_coordinate_descent(norm_tr_f_matrix, tr_output,
		init_weights, l1₂, tol)
	(size(weights1e8), typeof(weights1e8))
end

# ╔═╡ c07b70a0-755f-11eb-30ed-cba73d960ff5
md"""
**Quiz Question**

What features had non-zero weight in this case?

```julia
features_nzw(a_features, weights1e8)
```
"""

# ╔═╡ 81964a24-7562-11eb-3ff3-4370b4056f74
with_terminal() do
	for t ∈ features_nzw(a_features, weights1e8)
		@printf("%-20s => %2.5e / %15.5f\n", string(t[1]), t[2], t[2])
	end
end

# ╔═╡ c05c69c6-755f-11eb-0442-ef31019d90ff
md"""
##### Learn with l₁ of 1e4
"""

# ╔═╡ 98a45748-7561-11eb-154a-35b7dd014d41
begin
	l1₃=1e4
	# init_weights₃ = zeros(length(a_features) + 1)
	# tol = 1.0

	weights1e4 = lasso_cyclical_coordinate_descent(norm_tr_f_matrix, tr_output,
		init_weights, l1₃, tol)
	(size(weights1e4), typeof(weights1e4))
end

# ╔═╡ 9856bf2e-7561-11eb-0887-ff4f836e557e
md"""
**Quiz Question**

What features had non-zero weight in this case?

```julia
features_nzw(a_features, weights1e4)
```
"""

# ╔═╡ 08a08158-7563-11eb-0141-15cbef1ee9a6
with_terminal() do
	for t ∈ features_nzw(a_features, weights1e4)
		@printf("%-20s => %2.5e / %15.5f\n", string(t[1]), t[2], t[2])
	end
end

# ╔═╡ a7b5fba4-7563-11eb-2a24-f34b6296b447
md"""
#### Rescaling learned weights


Recall that we normalized our feature matrix, before learning the weights.  To use these weights on a test set, we must normalize the test data in the same way.

Alternatively, we can rescale the learned weights to include the normalization, so we never have to worry about normalizing the test data: 

In this case, we must scale the resulting weights so that we can make predictions with *original* features:
 1. Store the norms of the original features to a vector called `norms`:
```
features, norms = normalize_features(features)
```
 2. Run Lasso on the normalized features and obtain a `weights` vector
 3. Compute the weights for the original features by performing element-wise division, i.e.
```
weights_normalized = weights / norms
```
Now, we can apply `weights_normalized` to the test data, without normalizing it!
"""

# ╔═╡ c088e8d0-7563-11eb-0d5c-79ed7f100599
## Create a normalized version of each of the weights learned above. (weights1e4, weights1e7, weights1e8).

begin
all_weights = [
		w ./ vec(norms_tr) for w ∈ (weights1e4, weights1e7, weights1e8)
];
	
size(all_weights)
end

# ╔═╡ c048f716-7563-11eb-214f-35c773b47c1d
md"""
#### Evaluating each of the learned models on the test data

Let's now evaluate the three models on the test data
"""

# ╔═╡ c02e5654-7563-11eb-390b-ef4352bcf1fa
(test_f_matrix, test_output) = get_data(test_data, a_features, :price);

# ╔═╡ 61567f18-7564-11eb-2301-9dbec94cf157
md"""
Compute the RSS of each of the three normalized weights on the (unnormalized) `test_feature_matrix`
"""

# ╔═╡ 9298e410-7566-11eb-0b86-5376fcf9eabc
md"""
**Quiz Question**

Which model performed best on the test data?
"""

# ╔═╡ 611cc566-7564-11eb-168f-052aa0209973
with_terminal() do
	best_rss, best_weights = (nothing, nothing)
	w_names = (:weights1e4, :weights1e7, :weights1e8)
	
	for (weights, label) ∈ zip(all_weights, w_names)
  		rss = calc_rss(test_f_matrix, test_output, weights)
  		@printf("for %-10s => rss: %1.5e\n", label, rss)
		
  		if isnothing(best_rss) || rss < best_rss
    		best_rss, best_weights = rss, label
		end
	end
	
	@printf("\nbest model/weights: for %-10s => rss: %1.5e\n", best_weights, best_rss)
end

# ╔═╡ 6e03f5ea-7564-11eb-3309-9f6131c552cf


# ╔═╡ Cell order:
# ╟─5b0c59ae-753d-11eb-1a8a-6fa0bf99a357
# ╠═a450c12c-753d-11eb-06fc-29c5d907cb06
# ╟─7e12df34-7549-11eb-30a5-b3099a8f77f6
# ╠═9e5e1060-7549-11eb-0418-55bcd0b8c757
# ╠═a4ce9686-7549-11eb-2172-2d47391d1260
# ╟─2fa16676-754a-11eb-00cc-896d0e28b246
# ╟─4e5d02be-754a-11eb-061f-41ab2f747c54
# ╠═dc99317e-754a-11eb-0a8e-61ec5dff0507
# ╠═e9128af8-7555-11eb-1f4d-d1e17d70f4b3
# ╠═0d7f1fa0-754c-11eb-330b-d31a86f5566c
# ╟─984c778a-7557-11eb-21f8-dd2502fe7b73
# ╟─eb96bfc8-7556-11eb-2045-0f0e7adabe76
# ╠═9c8d9610-754f-11eb-05c3-edd271661b84
# ╠═eb872178-754f-11eb-177d-7bb5ee1046f5
# ╠═20799d0c-7550-11eb-16ee-676be88a80af
# ╟─06be0f04-7557-11eb-3f7c-bfc5e5a172d4
# ╠═06a0f408-7557-11eb-182e-8312809145d1
# ╟─06814074-7557-11eb-2359-8fa03900e3f8
# ╠═fcc38dfc-7557-11eb-048c-93e95ce9bbbc
# ╠═04b4ddf8-7557-11eb-0d69-f7a89e8936f8
# ╟─0498db3c-7557-11eb-1af6-9de231a4256f
# ╟─0480646e-7557-11eb-3aa9-af64a4502ff9
# ╠═046a8ad4-7557-11eb-2aa4-e37c7f37f711
# ╟─044fc410-7557-11eb-3a80-eb215ebc3f5f
# ╠═0434c2be-7557-11eb-0592-1b733cae76ca
# ╟─041c6f52-7557-11eb-050e-c97dee4f3220
# ╟─4f9ee984-755a-11eb-06cb-8da12ef68907
# ╠═53782228-755a-11eb-1554-394795823745
# ╠═535bf7b0-755a-11eb-06ea-8b0e0baec03c
# ╟─0403eec8-7557-11eb-18f7-13b0403e0a5b
# ╠═03ec653c-7557-11eb-2240-8b060cbb17af
# ╠═dcf8bdc2-755b-11eb-04da-39f89dd5709c
# ╠═03d065a8-7557-11eb-1cf0-add64e6d7596
# ╟─cf49001e-755e-11eb-33d9-9182f04696f8
# ╠═d35515f6-755e-11eb-3da0-0bc64c7ed3b1
# ╠═acd4c536-755e-11eb-1b0c-9573130238dc
# ╟─a941a03a-755f-11eb-38f7-455aaa090d7f
# ╠═c26ef2fe-755f-11eb-28e4-ef66e958b894
# ╠═c251d9bc-755f-11eb-2fe2-13c496a0abbd
# ╟─c0ff7904-755f-11eb-2ad7-572acb587b6f
# ╠═c0e5f506-755f-11eb-1ad1-6906e5ccfd85
# ╟─c0ca4482-755f-11eb-156b-a7011072410a
# ╠═230c74ce-7562-11eb-12d7-43e526f2ec26
# ╟─c0b05b74-755f-11eb-319e-9f5eff250eeb
# ╠═c095ab50-755f-11eb-2347-79fca41a8555
# ╟─c07b70a0-755f-11eb-30ed-cba73d960ff5
# ╠═81964a24-7562-11eb-3ff3-4370b4056f74
# ╟─c05c69c6-755f-11eb-0442-ef31019d90ff
# ╠═98a45748-7561-11eb-154a-35b7dd014d41
# ╟─9856bf2e-7561-11eb-0887-ff4f836e557e
# ╠═08a08158-7563-11eb-0141-15cbef1ee9a6
# ╟─a7b5fba4-7563-11eb-2a24-f34b6296b447
# ╠═c088e8d0-7563-11eb-0d5c-79ed7f100599
# ╟─c048f716-7563-11eb-214f-35c773b47c1d
# ╠═c02e5654-7563-11eb-390b-ef4352bcf1fa
# ╟─61567f18-7564-11eb-2301-9dbec94cf157
# ╟─9298e410-7566-11eb-0b86-5376fcf9eabc
# ╠═611cc566-7564-11eb-168f-052aa0209973
# ╠═6e03f5ea-7564-11eb-3309-9f6131c552cf
