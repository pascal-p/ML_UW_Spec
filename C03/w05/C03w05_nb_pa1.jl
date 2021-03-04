### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 77311686-7a2b-11eb-2bc7-95b2a2211969
begin
	using Pkg
	Pkg.activate("MLJ_env", shared=true)
	
	using MLJ
	using CSV
	using DataFrames
	using PlutoUI
	using Random
	using Test
	using Printf
	using Plots
end

# ╔═╡ 142f3778-7c71-11eb-2739-4d408a8309e2
using Distributions

# ╔═╡ e9f8e942-7b29-11eb-1d64-d7e7ff02159c
begin
	include("./utils.jl");
	include("./dt_utils.jl");
end

# ╔═╡ 5435ff66-7a2b-11eb-1474-b96dcad21315
md"""
## C03w05: Exploring Ensemble Methods

In this assignment, we will explore the use of boosting. We will use the pre-implemented gradient boosted trees in Turi Create. We will:

  - Use SFrames to do some feature engineering.
  - Train a boosted ensemble of decision-trees (gradient boosted trees) on the LendingClub dataset.
  - Predict whether a loan will default along with prediction probabilities (on a validation set).
  - Evaluate the trained model and compare it with a baseline.
  - Find the most positive and negative loans using the learned model.
  - Explore how the number of trees influences classification performance.
"""

# ╔═╡ 9e476ba0-7c5e-11eb-0082-6d4ffa97add4
const DF = Union{DataFrame, SubDataFrame}

# ╔═╡ 771130dc-7a2b-11eb-170f-934c11fceede
md"""
#### Load LendingClub dataset
"""

# ╔═╡ 76f62e40-7a2b-11eb-095e-f36e207f06a2
begin
	loans =  CSV.File(("../../ML_UW_Spec/C03/data/lending-club-data.csv"); 
	header=true) |> DataFrame;
	first(loans, 3)
end

# ╔═╡ 76de16d4-7a2b-11eb-09c6-e36b6c893c1f
md"""
Like the previous assignment, we reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.
"""

# ╔═╡ 76c17c4a-7a2b-11eb-2c47-97106bd58f99
insertcols!(loans, :safe_loans => ifelse.(loans.bad_loans .== 0, 1, -1),
 	makeunique=true);

# ╔═╡ 76a83684-7a2b-11eb-1961-cb133db8c164
md"""
#### Selecting Features

In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are **described in the code comments** below. If you are a finance geek, the [LendingClub](https://www.lendingclub.com/) website has a lot more details about these features.

The features we will be using are described below.
"""

# ╔═╡ 768d6b08-7a2b-11eb-388a-8763c8da0a51
begin
	const Features = [
		:grade,                     # grade of the loan (categorical)
		:sub_grade_num,             # sub-grade of the loan as a number from 0 to 1
		:short_emp,                 # one year or less of employment
		:emp_length_num,            # number of years of employment
		:home_ownership,            # home_ownership status: own, mortgage or rent
		:dti,                       # debt to income ratio
		:purpose,                   # the purpose of the loan
		:payment_inc_ratio,         # ratio of the monthly payment to income
		:delinq_2yrs,               # number of delinquincies 
		:delinq_2yrs_zero,          # no delinquincies in last 2 years
		:inq_last_6mths,            # number of creditor inquiries in last 6 months
		:last_delinq_none,          # has borrower had a delinquincy
		:last_major_derog_none,     # has borrower had 90 day or worse rating
		:open_acc,                  # number of open credit accounts
		:pub_rec,                   # number of derogatory public records
		:pub_rec_zero,              # no derogatory public records
		:revol_util,                # percent of available credit being used
		:total_rec_late_fee,        # total late fees received to day
		:int_rate,                  # interest rate of the loan
		:total_rec_int,             # interest received to date
		:annual_inc,                # annual income of borrower
		:funded_amnt,               # amount committed to the loan
		:funded_amnt_inv,           # amount committed by investors for the loan
		:installment,               # monthly payment owed by the borrower
  	]

	const Features_ST = Dict{Symbol, DataType}(
    	:grade => OrderedFactor{7},          # because 7 ≠ values
    	:home_ownership => Multiclass{4},
    	:purpose => Multiclass{12},
		
		:last_delinq_none => OrderedFactor{2},
		:last_major_derog_none => OrderedFactor{2},
		:delinq_2yrs_zero => OrderedFactor{2},
		:pub_rec_zero => OrderedFactor{2},
		
		## using autotype for the rest
  	)
	
	const Target = :safe_loans # prediction (y) (+1 means safe, -1 is risky)

	const Target_ST = Dict{Symbol, DataType}(
    	Target => Multiclass{2},
  	)

	## Considered categorical features
	const Cat_Features = [
		keys(Features_ST)..., 
		:sub_grade_num, 
		:short_emp, 
		:emp_length_num,
		:delinq_2yrs,
		:inq_last_6mths,
		:open_acc,
		:pub_rec, 
		:annual_inc, 
		:funded_amnt,
		:funded_amnt_inv,
	] |> unique
	
  	## Extract the feature columns and Target column
 	select!(loans, [Features..., Target]);
 	length(names(loans)), names(loans)
end

# ╔═╡ 47382678-7bec-11eb-34ff-9b3613ca036c
md"""
#### Skipping observations with missing values

Recall from the lectures that one common approach to coping with missing values is to skip observations that contain missing values.
"""

# ╔═╡ 56d0dfee-7bec-11eb-2890-7577dda5d4de
begin
	s₀ = size(loans)[1];
	dropmissing!(loans);
	s₁ = size(loans)[1];
	## how many missing row were dropped:
	s₀ - s₁ 
end

# ╔═╡ 37575388-7c9f-11eb-1263-8d4dc28d711b
length(Cat_Features), Cat_Features

# ╔═╡ e73da3ea-7c92-11eb-3169-f5b8a11e4154
# reduce(vcat, loans[!, :pub_rec_zero]) |> unique

# ╔═╡ 7673d8e4-7a2b-11eb-02b4-2f837181b4e9
md"""
#### Subsample dataset to make sure classes are balanced

Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points.
"""

# ╔═╡ ef0b9e82-7c61-11eb-2cdf-2bc585f76e61
begin
	function loan_ratios(safe_, risky_; n=size(loans)[1])
  		(size(safe_)[1] / n, size(risky_)[1] / n)
	end
	
	function partition_sr(df::DF; target=Target)
		"""Partition between safe and risky loan"""
		(df[df[!, target] .== 1, :], df[df[!, target] .!= 1, :])  # was -1
	end
	
	function resample!(safe_loans::DF, risky_loans::DF)
		"""
		Since there are fewer risky loans than safe loans, find the ratio of 
		the sizes and use that percentage to downsample the safe loans.
		"""
		## => deps on sampling(...)
		n = size(safe_loans)[1]
		perc = size(risky_loans)[1] / n
		
		## down sample safe loans
		safe_loans = safe_loans[sampling(n, perc; seed=42), :];   
		[risky_loans; safe_loans] 
	end
	
end

# ╔═╡ 542db682-7a2e-11eb-35ad-25c82d6919e0
begin
	safe_loans₀, risky_loans₀ = partition_sr(loans)
	p_safe_loans₀, p_risky_loans₀ = loan_ratios(safe_loans₀, risky_loans₀);
	
	with_terminal() do
		@printf("Number of safe loans  : %d\n", size(safe_loans₀)[1])
		@printf("Number of risky loans : %d\n", size(risky_loans₀)[1])
		@printf("Percentage of safe loans:  %1.2f%%\n", p_safe_loans₀ * 100.) 
		@printf("Percentage of risky loans: %3.2f%%\n", p_risky_loans₀ * 100.)
	end
end

# ╔═╡ 4f4034e6-7c5f-11eb-0530-b3e9f0d7cc94
loans_data = resample!(safe_loans₀, risky_loans₀);

# ╔═╡ 762690b8-7a2b-11eb-0257-f5a54932adc5
begin
	safe_loans₁, risky_loans₁ = partition_sr(loans_data)
	p_safe_loans₁, p_risky_loans₁ = loan_ratios(safe_loans₁, risky_loans₁;
		n=size(loans_data)[1]);
	
	with_terminal() do
		@printf("Number of safe loans  : %d\n", size(safe_loans₁)[1])
		@printf("Number of risky loans : %d\n", size(risky_loans₁)[1])
		@printf("Percentage of safe loans:  %1.2f%%\n", p_safe_loans₁ * 100.) 
		@printf("Percentage of risky loans: %3.2f%%\n", p_risky_loans₁ * 100.)
	end
end

# ╔═╡ 3956455a-7c62-11eb-0ed6-cb2cda3ca879
md"""
##### Deal with Categorical features

initial mapping:
"""

# ╔═╡ 9ca8a008-7cad-11eb-1fdc-c31031290a79
with_terminal() do
	for n ∈ names(loans_data)
		@printf("%-25s => %50s\n", n, scitype(loans_data[!, n]))
	end
end

# ╔═╡ e1245fd0-7bf2-11eb-071c-0fb7dd06e885
begin
	## Scientific Type coercion
	coerce_map = [Features_ST..., Target_ST...]
	coerce!(loans_data, coerce_map...);
	coerce_map
end

# ╔═╡ a67aaa04-7cad-11eb-2a40-8585bad43994
with_terminal() do
	for n ∈ names(loans_data)
		@printf("%-25s => %50s\n", n, scitype(loans_data[!, n]))
	end
end

# ╔═╡ e209e38c-7c94-11eb-09fb-279bcffec0f7
md"""

For a number of the remaining features which are treated as Count there are few unique values in which case it might make more sense to recode them as OrderedFactor, this can be done with autotype.
"""

# ╔═╡ 9441f1d0-7c94-11eb-139d-67562ee93876
coerce!(loans_data, autotype(loans_data, :few_to_finite));

# ╔═╡ 9e7aa038-7c9f-11eb-32e9-b3579071450e
with_terminal() do
	for n ∈ names(loans_data)
		@printf("%-25s => %50s\n", n, scitype(loans_data[!, n]))
	end
end

# ╔═╡ 643ee9de-7bf4-11eb-1d64-d7e7ff02159c
md"""
### MLJ AdaBoostStumpClassifier
"""

# ╔═╡ f3999fe6-7c69-11eb-14c2-755155d7fe83
function hot_encode_fn(df::DF, cat_features)
	"""
	hot-encoding of categorical features
	"""
	ohe_mdl = OneHotEncoder(; 
		features=cat_features, 
		ordered_factor=false, 
		ignore=false
	)
	
	ohe_mach = machine(ohe_mdl, df)
	fit!(ohe_mach)
	MLJ.transform(ohe_mach, df)  ## return df hotencoded
end

# ╔═╡ 72b7ce48-7caf-11eb-30bb-6561f21e7dcf
with_terminal() do
	for n ∈ names(loans_data)
		@printf("%-20s => %-60s\n", n, typeof(loans_data[!, n]))
	end
end

# ╔═╡ 97dcb9c4-7bf4-11eb-071c-0fb7dd06e885
begin
	attrs_to_ohe = Cat_Features 
	loans_df_ohe = hot_encode_fn(loans_data, attrs_to_ohe);
	length(names(loans_df_ohe)), names(loans_df_ohe)
end

# ╔═╡ 7d4a709a-7caf-11eb-0666-893547dad910
with_terminal() do
	for n ∈ names(loans_df_ohe)
		@printf("%-25s => %-60s\n", n, typeof(loans_df_ohe[!, n]))
	end
end

# ╔═╡ a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
md"""
#### Train-Validation split

We split the data into a train/validation split with 80% of the data in the training set and 20% of the data in the validation set. 
"""

# ╔═╡ 6fad8afe-7a33-11eb-3208-6b289c3ccd23
begin
	train_data, validation_data = train_test_split(loans_df_ohe; split=0.8, seed=42);
	size(train_data), size(validation_data)
end

# ╔═╡ a5ff6eea-7c62-11eb-2cdf-2bc585f76e61
begin
	attrs = [Symbol("safe_loans__-1"), :safe_loans__1]
	
	Xₜ = select(train_data, Not(Target)) # Not(attrs))
	yₜ = select(train_data, Target)
	yₜ = yₜ[:, Target];   ## cast to vector
end

# ╔═╡ f6c23cf6-7cae-11eb-042d-1153a4f16773
with_terminal() do
	for n ∈ names(Xₜ)
		@printf("%-25s => %50s\n", n, scitype(Xₜ[!, n]))
	end
end

# ╔═╡ 14d15ed4-7caf-11eb-2cfe-c91267cec94f
with_terminal() do
	# for n ∈ names(Xₜ)
		@printf("%-20s => %50s\n", Target, scitype(yₜ))
	# end
end

# ╔═╡ e018a54a-7c5d-11eb-34ff-9b3613ca036c
@load AdaBoostStumpClassifier

# ╔═╡ 79ac6bc6-7c84-11eb-3e1d-515509843831
function abs_clf(X, y; n_iter=5)
	ada_clf = MLJDecisionTreeInterface.AdaBoostStumpClassifier(n_iter=n_iter)
	ada_mach = machine(ada_clf, X, y)
	fit!(ada_mach, rows=1:size(X)[1])
	ada_mach
end

# ╔═╡ 5c4a9dee-7bff-11eb-34ff-9b3613ca036c
ada_mach = abs_clf(Xₜ, yₜ; n_iter=5)

# ╔═╡ 322503ba-7c5e-11eb-3374-fb3932c68191
fitted_params(ada_mach)

# ╔═╡ edbb6bb8-7c87-11eb-3124-8506ee0b6781
ya_accuracy(ŷ, y) = sum(ŷ .== y) / length(y)

# ╔═╡ 7ee1f0dc-7c86-11eb-204d-4726e4d62fd4
# begin
	# evaluate!(ada_mach,
	#           resampling=Holdout(fraction_train=0.7, shuffle=true, rng=1234),
	#           measure=[Accuracy()],
	# 		    check_measure=false)
	# evaluate!(ada_mach, resampling=Holdout(), measure=[ya_accuracy],
	# 	operation=predict, 	verbosity=1)

	# r = evaluate!(ada_mach, resampling=CV(nfolds=3), measure=[ya_accuracy], 
	# 	operation=predict, 	verbosity=0)
	# (r.measure, r.measurement)
# end

# ╔═╡ 7e79f75c-7c63-11eb-0429-374262cb7f0d
md"""
#### Making predictions

Let us consider a few positive and negative examples from the validation set. We will do the following:

  - Predict whether or not a loan is likely to default.
  - Predict the probability with which the loan is likely to default.
"""

# ╔═╡ b3a8e8b6-7c63-11eb-2254-d1728a2f197d
begin
  	v_safe_loans, v_risky_loans = partition_sr(validation_data; target=Target)
	sample_valid_data = vcat(v_safe_loans[1:2, :], v_risky_loans[1:2, :])
end

# ╔═╡ ddf3fcec-7c6c-11eb-0530-b3e9f0d7cc94
function do_predict(mach, Xₙ, yₙ)
	"""
	Note:
	using sample_valid_data:
	mode is: [1, 1, -1, 1]
	proba:   [0.606493, 0.542905, 0.606493, 0.393507]
	
	raw_proba (per level -1 / 1)
	  UnivariateFinite{ScientificTypes.Multiclass{2}}(-1=>0.394, 1=>0.606)
      UnivariateFinite{ScientificTypes.Multiclass{2}}(-1=>0.457, 1=>0.543)
	  UnivariateFinite{ScientificTypes.Multiclass{2}}(-1=>0.606, 1=>0.394)
      UnivariateFinite{ScientificTypes.Multiclass{2}}(-1=>0.394, 1=>0.606)
		
	this means that the probas for predicting [1, 1, -1, 1] are:
		[0.606493, 0.542905, 0.606493, 0.606493]  # see how last oen differs from proabs given above...
	
	Theerefore here, I am going to use for the proba, the following:  
	  eachrow(pdf(ŷ_probs, levels(yₙ))) |> r -> maximum.(r)
	
	"""
	ŷ_probs = predict(ada_mach, Xₙ) 
	ŷ = broadcast(mode, ŷ_probs)
	
	ratio_corr_pred = sum(ŷ .== yₙ) / size(Xₙ)[1]
	proba = eachrow(pdf(ŷ_probs, levels(yₙ))) |> r -> maximum.(r)
	
	# used: proba=broadcast(pdf, ŷ_probs, yₙ) which is not what I am expecting...
	(mode=ŷ, raw_proba=ŷ_probs, proba=proba, 
		ratio_corr=ratio_corr_pred, pdf_per_level=pdf(ŷ_probs, levels(yₙ))) 
end

# ╔═╡ 73b7d41c-7c66-11eb-224b-1d7474295bd2
begin
	# using Distributions
	Xsv = select(sample_valid_data, Not(Target))
	ysv = sample_valid_data.safe_loans
	
	predsv = do_predict(ada_mach, Xsv, ysv)
	
	with_terminal() do
		println("Predictions   mode: $(predsv.mode)")
		println("Predictions probas: $(round.(predsv.proba, digits=4))")
		# println("Predictions probas/level: $(predsv.pdf_per_level)")
		println("Ratio of correct predictions: $(round(predsv.ratio_corr, digits=4))")
	end
end

# ╔═╡ dfe9ab1e-7c71-11eb-21e1-7da6f486bb75
predsv.mode

# ╔═╡ e805a514-7c71-11eb-13cd-a73039c33692
predsv.raw_proba

# ╔═╡ 0773fae6-7c71-11eb-2638-8fe05034ba2d
predsv.pdf_per_level

# ╔═╡ c61a11c8-7c72-11eb-246e-15a9c4f4cb68
eachrow(predsv.pdf_per_level) |> r -> maximum.(r)

# ╔═╡ f093cc04-7c6e-11eb-1d27-2173c9ebeab7
## least likely to be a safe loan
begin
	ix_least = argmin(predsv.proba) 
	(ix_least, round(predsv.proba[ix_least], digits=4))
end

# ╔═╡ 3c4f3a96-7c6c-11eb-0082-6d4ffa97add4
md"""
**Quiz Question**: What percentage of the predictions on sample_validation_data did model_5 get correct?
  - 75% (cf. above cell)

**Quiz Question**: According to our model, which loan is the LEAST likely to be a safe loan?

  - the second one

**Checkpoint**: Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1?
  - Ok see below.
"""

# ╔═╡ c7a5de14-7c75-11eb-1c8e-d718788cb40c
all(p -> p ≥ 0.5, predsv.proba[predsv.mode .== 1])

# ╔═╡ bf1ff544-7c6e-11eb-37f2-09f2102bbc7b
md"""
#### Evaluating the model on the validation data
"""

# ╔═╡ e27ffc5a-7c96-11eb-23b0-470fe22ceaea
begin
	Xᵥ = select(validation_data, Not(Target))
	yᵥ = select(validation_data, Target)
	yᵥ = yᵥ[:, Target];   ## cast to vector
end

# ╔═╡ 6d9cc902-7c70-11eb-3b71-ff1408d9d504
begin
	## on whole validation data	
	preds = do_predict(ada_mach, Xᵥ, yᵥ)
	lim = 8
	
	with_terminal() do
		println("Predictions   mode: $(preds.mode[1:lim])...")
		println("Predictions probas: $(round.(preds.proba[1:lim], digits=4))")
		println("Ratio of correct predictions: $(round(preds.ratio_corr, digits=4))")
	end
end

# ╔═╡ 41919468-7c76-11eb-3389-471f19f9e92f
md"""
Calculate the number of *false positives* and the the number of *false negatives* made by the model where:

  - *false positives* are loans that were predicted to be safe but were actually risky
  - *false negatives* are loans that were were predicted to be risky but were actually safe.  

"""

# ╔═╡ f698c078-7c76-11eb-0841-37338a78bd2c

fp = (preds.mode .== 1) |>                          ## predicted safe == 1
  ba -> (validation_data.safe_loans[ba] .== -1) |>  ## but actually risky == -1
  sum

# ╔═╡ 4bd262b2-7c77-11eb-310a-e1f639f135d5
fn = (preds.mode .== -1) |>                          ## predicted riksy == -1
  ba -> (validation_data.safe_loans[ba] .== 1) |>    ## but actually safe == 1
  sum	

# ╔═╡ 9f9470a6-7c78-11eb-006f-8b06746cbc01
md"""
Using the costs defined above and the number of false positives and false negatives for the decision tree, we can calculate the total cost of the mistakes made by the decision tree model as follows:

cost = $10,000 * 1936  + $20,000 * 1503 = $49,420,000

The total cost of the mistakes of the model is $49.42M. That is a lot of money!.

**Quiz Question**: Using the same costs of the false positives and false negatives, what is the cost of the mistakes made by the boosted tree model (model_5) as evaluated on the validation_set?
"""

# ╔═╡ 9f6eb654-7c78-11eb-1a81-9ba2e6a0cd6d
cost = fp * 20000 + fn * 10000

# ╔═╡ c6261684-7c78-11eb-02b4-bb1911d035a7
md"""
#### Most positive & negative loans.

In this section, we will find the loans that are most likely to be predicted safe. We can do this in a few steps:

  1. Use the ada_mach (the model with 5 trees) and make probability predictions for all the loans in the validation dataset (*already done*).
  1. Add theses probability predictions as a column called predictions into the validation_data dataframe.
  1. Sort the data (in descreasing order) by the probability predictions.

"""

# ╔═╡ f9c2b1f2-7c80-11eb-333e-1da96d113d79
begin
	valid_ext_data = hcat(validation_data, preds.proba, makeunique=true)
	rename!(valid_ext_data, :x1 => :predictions)
	
	first(valid_ext_data, 3)
end

# ╔═╡ 9f01203c-7c80-11eb-0666-893547dad910
## Top 7 predictions
with_terminal() do
	@show sort(valid_ext_data, :predictions, rev=true)[1:7, [:grade, Target, :predictions]]
end

# ╔═╡ 9ee273a8-7c80-11eb-30bb-6561f21e7dcf
## bottom 7 predictions
with_terminal() do
	@show sort(valid_ext_data, :predictions, rev=false)[1:7, [:grade, Target, :predictions]]
end

# ╔═╡ d5d5c1f0-7c83-11eb-0987-bfc19d4850e3
md"""

### Effect of adding more trees

In this assignment, we will train 5 different ensemble classifiers in the form of gradient boosted trees. We will train models with 10, 50, 100, 200, and 500 trees. We use the n_iter parameter in AdaBoostStumpClassifier.

"""

# ╔═╡ 005e87fe-7c84-11eb-01f1-8fddeb393779
const N_ITERS = [10, 50,  75, 100, 125, 150, 200, 250, 300]

# ╔═╡ 0ca7bc74-7c84-11eb-252c-3bf3df6b6635
function try_models(X, y)
	hsh = Dict{Symbol, Any}()
	
  	for (ix, n_iter) ∈ enumerate(N_ITERS)
		sym = Symbol("model_", ix)
		hsh[sym] = Dict{Symbol, Any}(:ada_model => nothing, :n_iter => n_iter)
    	hsh[sym][:ada_model] = abs_clf(X, y; n_iter)
	end
	
	hsh
end

# ╔═╡ f9ff756a-7c85-11eb-0b45-0f1b3735844f
function eval_acc_model(mach, X, y)
	ŷ = predict_mode(mach, X) 
	sum(ŷ .== y) / length(y)
end

# ╔═╡ 7dccb71e-7c8a-11eb-175d-0fc5400512d8
hsh = try_models(Xₜ, yₜ); # using the training data...

# ╔═╡ b33f43b2-7c85-11eb-21e7-f30c1eeb4785
begin
	## On vaLidation data
	best_model, best_acc = nothing, nothing
	m_acc = []
	
	for m ∈ keys(hsh)
  		acc = eval_acc_model(hsh[m][:ada_model], Xᵥ, yᵥ)
		push!(m_acc, acc)
  		
  		if isnothing(best_acc) || acc > best_acc
    		global best_model = m
    		global best_acc = acc
		end
	end
	
	with_terminal() do
		for (ix, acc) ∈ enumerate(m_acc)
			@printf("- model_%d - validation accuracy %1.4f\n", 
				ix, round(acc, digits=4))
		end
		@printf("\n+ best: %7s - validation accuracy %1.4f / max iter: %3d\n",
			best_model, round(best_acc, digits=4), hsh[best_model][:n_iter])
	end
end

# ╔═╡ 662e0a14-7c8e-11eb-245a-07169e1831ff
md"""
**Quiz Question**: Which model has the best accuracy on the validation_data?
  - model 4 (200 iter)

**Quiz Question**: Is it always true that the model with the most trees will perform best on test data?
  - No

"""

# ╔═╡ b9c2ca34-7c8e-11eb-1fa5-2d491d1dd7c6
begin
	train_errors, valid_errors = Float64[], Float64[]

	for m ∈ keys(hsh)
  		train_err = 1. - eval_acc_model(hsh[m][:ada_model], Xₜ, yₜ)
 		valid_err = 1. - eval_acc_model(hsh[m][:ada_model], Xᵥ, yᵥ)
  		push!(train_errors, train_err)
  		push!(valid_errors, valid_err)
	end
	
	train_errors, valid_errors
end

# ╔═╡ 98ba3708-7c90-11eb-0ba5-01dad28c3126
begin
	plot(N_ITERS, train_errors, color=[:lightblue], linewidth=3,
		label="train_error", leg=:topright) 
	plot!(N_ITERS, valid_errors, color=[:orange], linewidth=3,
		label="valid_error", leg=:topright)
	plot!(ylims=(0.25, 0.45))
end

# ╔═╡ Cell order:
# ╟─5435ff66-7a2b-11eb-1474-b96dcad21315
# ╠═77311686-7a2b-11eb-2bc7-95b2a2211969
# ╠═9e476ba0-7c5e-11eb-0082-6d4ffa97add4
# ╟─771130dc-7a2b-11eb-170f-934c11fceede
# ╠═76f62e40-7a2b-11eb-095e-f36e207f06a2
# ╟─76de16d4-7a2b-11eb-09c6-e36b6c893c1f
# ╠═76c17c4a-7a2b-11eb-2c47-97106bd58f99
# ╠═e9f8e942-7b29-11eb-1d64-d7e7ff02159c
# ╟─76a83684-7a2b-11eb-1961-cb133db8c164
# ╠═768d6b08-7a2b-11eb-388a-8763c8da0a51
# ╟─47382678-7bec-11eb-34ff-9b3613ca036c
# ╠═56d0dfee-7bec-11eb-2890-7577dda5d4de
# ╠═37575388-7c9f-11eb-1263-8d4dc28d711b
# ╠═e73da3ea-7c92-11eb-3169-f5b8a11e4154
# ╟─7673d8e4-7a2b-11eb-02b4-2f837181b4e9
# ╠═ef0b9e82-7c61-11eb-2cdf-2bc585f76e61
# ╠═542db682-7a2e-11eb-35ad-25c82d6919e0
# ╠═4f4034e6-7c5f-11eb-0530-b3e9f0d7cc94
# ╠═762690b8-7a2b-11eb-0257-f5a54932adc5
# ╟─3956455a-7c62-11eb-0ed6-cb2cda3ca879
# ╠═9ca8a008-7cad-11eb-1fdc-c31031290a79
# ╠═e1245fd0-7bf2-11eb-071c-0fb7dd06e885
# ╠═a67aaa04-7cad-11eb-2a40-8585bad43994
# ╟─e209e38c-7c94-11eb-09fb-279bcffec0f7
# ╠═9441f1d0-7c94-11eb-139d-67562ee93876
# ╠═9e7aa038-7c9f-11eb-32e9-b3579071450e
# ╟─643ee9de-7bf4-11eb-1d64-d7e7ff02159c
# ╠═f3999fe6-7c69-11eb-14c2-755155d7fe83
# ╠═72b7ce48-7caf-11eb-30bb-6561f21e7dcf
# ╠═97dcb9c4-7bf4-11eb-071c-0fb7dd06e885
# ╠═7d4a709a-7caf-11eb-0666-893547dad910
# ╟─a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
# ╠═6fad8afe-7a33-11eb-3208-6b289c3ccd23
# ╠═a5ff6eea-7c62-11eb-2cdf-2bc585f76e61
# ╠═f6c23cf6-7cae-11eb-042d-1153a4f16773
# ╠═14d15ed4-7caf-11eb-2cfe-c91267cec94f
# ╠═e018a54a-7c5d-11eb-34ff-9b3613ca036c
# ╠═79ac6bc6-7c84-11eb-3e1d-515509843831
# ╠═5c4a9dee-7bff-11eb-34ff-9b3613ca036c
# ╠═322503ba-7c5e-11eb-3374-fb3932c68191
# ╠═edbb6bb8-7c87-11eb-3124-8506ee0b6781
# ╠═7ee1f0dc-7c86-11eb-204d-4726e4d62fd4
# ╟─7e79f75c-7c63-11eb-0429-374262cb7f0d
# ╠═b3a8e8b6-7c63-11eb-2254-d1728a2f197d
# ╠═142f3778-7c71-11eb-2739-4d408a8309e2
# ╠═ddf3fcec-7c6c-11eb-0530-b3e9f0d7cc94
# ╠═73b7d41c-7c66-11eb-224b-1d7474295bd2
# ╠═dfe9ab1e-7c71-11eb-21e1-7da6f486bb75
# ╠═e805a514-7c71-11eb-13cd-a73039c33692
# ╠═0773fae6-7c71-11eb-2638-8fe05034ba2d
# ╠═c61a11c8-7c72-11eb-246e-15a9c4f4cb68
# ╠═f093cc04-7c6e-11eb-1d27-2173c9ebeab7
# ╟─3c4f3a96-7c6c-11eb-0082-6d4ffa97add4
# ╠═c7a5de14-7c75-11eb-1c8e-d718788cb40c
# ╟─bf1ff544-7c6e-11eb-37f2-09f2102bbc7b
# ╠═e27ffc5a-7c96-11eb-23b0-470fe22ceaea
# ╠═6d9cc902-7c70-11eb-3b71-ff1408d9d504
# ╟─41919468-7c76-11eb-3389-471f19f9e92f
# ╠═f698c078-7c76-11eb-0841-37338a78bd2c
# ╠═4bd262b2-7c77-11eb-310a-e1f639f135d5
# ╟─9f9470a6-7c78-11eb-006f-8b06746cbc01
# ╠═9f6eb654-7c78-11eb-1a81-9ba2e6a0cd6d
# ╟─c6261684-7c78-11eb-02b4-bb1911d035a7
# ╠═f9c2b1f2-7c80-11eb-333e-1da96d113d79
# ╠═9f01203c-7c80-11eb-0666-893547dad910
# ╠═9ee273a8-7c80-11eb-30bb-6561f21e7dcf
# ╟─d5d5c1f0-7c83-11eb-0987-bfc19d4850e3
# ╠═005e87fe-7c84-11eb-01f1-8fddeb393779
# ╠═0ca7bc74-7c84-11eb-252c-3bf3df6b6635
# ╠═f9ff756a-7c85-11eb-0b45-0f1b3735844f
# ╠═7dccb71e-7c8a-11eb-175d-0fc5400512d8
# ╠═b33f43b2-7c85-11eb-21e7-f30c1eeb4785
# ╟─662e0a14-7c8e-11eb-245a-07169e1831ff
# ╠═b9c2ca34-7c8e-11eb-1fa5-2d491d1dd7c6
# ╠═98ba3708-7c90-11eb-0ba5-01dad28c3126
