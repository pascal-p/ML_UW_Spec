### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 90bdce42-7980-11eb-08ad-0f39e65e1864
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

# ╔═╡ 20f35fc2-799a-11eb-12e9-4588c897199d
include("utils.jl")

# ╔═╡ 51dfdce2-7980-11eb-229a-3b3196ded703
md"""
## Identifying safe loans with decision trees

The [LendingClub](https://www.lendingclub.com/) is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors. In this notebook, we will build a classification model to predict whether or not a loan provided by LendingClub is likely to [default](https://en.wikipedia.org/wiki/Default_%28finance%29).

In this notebook we will use data from the LendingClub to predict whether a loan will be paid off in full or the loan will be [charged off](https://en.wikipedia.org/wiki/Charge-off) and possibly go into default. In this assignment we will:

  - Use SFrames to do some feature engineering.
  - Train a decision-tree on the LendingClub dataset.
  - Predict whether a loan will default along with prediction probabilities (on a validation set).
  - Train a complex tree model and compare it to simple tree model.
"""

# ╔═╡ a36cad10-7980-11eb-212f-67774be5fd35
md"""
### Load LendingClub dataset

We will be using a dataset from the [LendingClub](https://www.lendingclub.com/). A parsed and cleaned form of the dataset is availiable [here](https://github.com/learnml/machine-learning-specialization-private). Make sure you **download the dataset** before running the following command.
"""

# ╔═╡ a36b25ee-7980-11eb-2308-e1b576d9883e
begin
	loans =  CSV.File(("../../ML_UW_Spec/C03/data/lending-club-data.csv"); 
	header=true) |> DataFrame;
	first(loans, 3)
end

# ╔═╡ a36b0642-7980-11eb-3a6f-e3f48bc58616
md"""
### Exploring some features

Let's quickly explore what the dataset looks like. First, let's print out the column names to see what features we have in this dataset.

"""

# ╔═╡ 19b49852-7981-11eb-1d65-99a8f9119422
length(names(loans)), names(loans)

# ╔═╡ 882c144c-7989-11eb-0e5d-4d2b0d13b459
function by_feature(df::DataFrame, feature::Symbol)
	groupby(df, feature) |> 
		df_ -> combine(df_, nrow => Symbol(feature, "_count"))
end

# ╔═╡ e84e8f1c-798e-11eb-23e6-0dafc1deaff4
function barplot(df::DataFrame, feature::Symbol)
	let df=by_feature(df, feature), fcnt=Symbol(feature, "_count"), yaxe=df[!, feature]
		
		bar(df[!, fcnt], yticks=(1:length(yaxe), collect(yaxe)), 
			orientation=:h, label=String(feature))
	end
end

# ╔═╡ d5331906-798f-11eb-21c5-bf17a158c639
barplot(loans, :grade)

# ╔═╡ 87ede65a-798d-11eb-2610-7db61052073b
md"""
We can see that over half of the loan grades are assigned values B or C. Each loan is assigned one of these grades, along with a more finely discretized feature called sub_grade (feel free to explore that feature column as well!). These values depend on the loan application and credit report, and determine the interest rate of the loan. More information can be found here.

Now, let's look at a different feature.
"""

# ╔═╡ 9405f590-798d-11eb-3a55-619e7d2c5cb6
barplot(loans, :sub_grade)

# ╔═╡ 93e75248-798d-11eb-2e4b-cb8072c75df7
barplot(loans, :home_ownership)

# ╔═╡ 93ce629a-798d-11eb-02fe-7b0556848660
md"""
This feature describes whether the loanee is mortaging, renting, or owns a home. We can see that a small percentage of the loanees own a home.
"""

# ╔═╡ 93b3c4aa-798d-11eb-04e9-f526070ac3eb
md"""
### Exploring the target column

The target column (label column) of the dataset that we are interested in is called `bad_loans`. In this column **1** means a *risky* (bad) loan **0** means a *safe* loan.

In order to make this more intuitive and consistent with the lectures, we reassign the target to be:
  - **+1** as a safe  loan, 
  - **-1** as a risky (bad) loan. 

We put this in a new column called `safe_loans`.

"""

# ╔═╡ 939cde84-798d-11eb-244a-9ba95ee1ada7
insertcols!(loans, :safe_loans => ifelse.(loans.bad_loans .== 0, 1, -1),
 	makeunique=true);

#insertcols!(loans, :safe_loans => ifelse.(loans.bad_loans .== 0, :ok, :ko),
#	makeunique=true);

# ╔═╡ 937f41c6-798d-11eb-0eec-ffd7cb58b2b6
md"""
Now, let us explore the distribution of the column safe_loans. This gives us a sense of how many safe and risky loans are present in the dataset.
"""

# ╔═╡ 9368145e-798d-11eb-2232-dd7572dfb755
barplot(loans, :safe_loans)

# ╔═╡ 9350b5f4-798d-11eb-2c9f-a1cbd7526d1a
with_terminal() do
	num_loans = size(loans)[1]

	println((num_safe_loans=round(sum(loans.safe_loans .== 1) / num_loans; digits=4), num_bad_loans=round(sum(loans.safe_loans .== -1) / num_loans; digits=4)))
end

# ╔═╡ 93194734-798d-11eb-3d96-1f8a46085885
md"""
We should have:
  - Around 81% safe loans
  - Around 19% risky loans

It looks like most of these loans are safe loans (thankfully). But this does make our problem of identifying risky loans challenging.
"""

# ╔═╡ 0120efaa-7992-11eb-313e-6b5e4dfd4800
md"""
### Features for the classification algorithm

In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are described in the code comments below. If you are a finance geek, the LendingClub website has a lot more details about these features.

"""

# ╔═╡ 1671f0ac-7992-11eb-20ff-912bf4353a24
begin
  const Features = [:grade,         # grade of the loan
            :sub_grade,             # sub-grade of the loan
            :short_emp,             # one year or less of employment
            :emp_length_num,        # number of years of employment
            :home_ownership,        # home_ownership status: own, mortgage or rent
            :dti,                   # debt to income ratio
            :purpose,               # the purpose of the loan
            :term,                  # the term of the loan
            :last_delinq_none,      # has borrower had a delinquincy
            :last_major_derog_none, # has borrower had 90 day or worse rating
            :revol_util,            # percent of available credit being used
            :total_rec_late_fee,    # total late fees received to day
           ]

	target = :safe_loans # prediction target (y) (+1 means safe, -1 is risky)

	# Extract the feature columns and target column
	select!(loans, [Features..., target])
	length(names(loans)), names(loans)
end

# ╔═╡ 16564a14-7992-11eb-3600-cd4cf1573933
md"""
What remains now is a subset of features and the target that we will use for the rest of this notebook. 
"""

# ╔═╡ 0aa86afe-79ff-11eb-0d39-4fee0ee8c7fc
md"""
#### MLJ Data Preparation

MLJ distinguished between (refer to [Data interpretation: Scientific Types](https://alan-turing-institute.github.io/DataScienceTutorials.jl/data/scitype/):
 - encoding or *machine type* (like an Integer, a String....) and 
 - how data should be interpreted or *scientific type*


Let's have a look at the current mapping machine type/scientific type on our dataframe. 
"""

# ╔═╡ 34c2bb96-79aa-11eb-1d64-d7e7ff02159c
schema(loans)

# ╔═╡ 67b8b802-79fa-11eb-241a-17e808148f7c
## which attribute/feature is Textual?

map(attr -> (attr, elscitype(loans[!, attr])), Features) |>
  ary -> filter(((_a, st) = t) -> st == Textual, ary)

# ╔═╡ e9c1bcc8-79fe-11eb-1437-9530a1eefbdb
md"""
Those textual attributes need to be coerced to more meaningful scitype. 

Based on the values of those attributes we can infer the folloing mapping:
"""

# ╔═╡ 1d6247c6-7a01-11eb-39e1-c1565af8a588
const Features_Scitype = Dict{}(
		:grade => OrderedFactor{7},          # because 7 ≠ values
		:sub_grade => OrderedFactor{35}, 
		:home_ownership => Multiclass{4},
		:purpose => Multiclass{12},
		:term => OrderedFactor{2},
	)

# ╔═╡ f2c7be06-7992-11eb-2c95-979c0965fe77
md"""
### Sample data to balance classes

As we explored above, our data is disproportionally full of safe loans. Let's create two datasets: one with just the safe loans (safe_loans_raw) and one with just the risky loans (risky_loans_raw).

"""

# ╔═╡ f29fef50-7992-11eb-2ed8-2bf706c9828a
begin
	safe_loans_raw = loans[loans[!, target] .== 1, :]
	risky_loans_raw = loans[loans[!, target] .== -1, :]

	with_terminal() do
		@printf("Number of safe loans  : %d\n", size(safe_loans_raw)[1])
		@printf("Number of risky loans : %d\n", size(risky_loans_raw)[1])
	end
end

# ╔═╡ 1629b576-7992-11eb-1fef-a3538166d17d
md"""
Now, write some code to compute below the percentage of safe and risky loans in the dataset and validate these numbers against what was given earlier in the assignment:
"""

# ╔═╡ 16055156-7992-11eb-3461-17b098dc3a27
begin
	function loan_ratios(safe_loans, risky_loans; n_loans=size(loans)[1])
  		(size(safe_loans)[1] / n_loans, size(risky_loans)[1] / n_loans)
	end

	p_safe_loans, p_risky_loans = loan_ratios(safe_loans_raw, risky_loans_raw)

	with_terminal() do
		@printf("Percentage of safe loans:  %1.2f%%\n", p_safe_loans * 100.) 
		@printf("Percentage of risky loans: %3.2f%%\n", p_risky_loans * 100.)
	end
end

# ╔═╡ 00f5c508-7992-11eb-1087-e7c63d454e06
md"""
One way to combat class imbalance is to undersample the larger class until the class distribution is approximately half and half. Here, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points. 

We used seed=42 so everyone gets the same results.
"""

# ╔═╡ 65959ca2-7996-11eb-2582-5f6f479ff6ff
randperm(MersenneTwister(42), 10)

# ╔═╡ b3fef5fa-7996-11eb-0d80-45603e97e1bb
randperm(MersenneTwister(42), 10)

# ╔═╡ b9cbc562-7996-11eb-1fb4-49c80af4690d
"""
Select a random sample perecentage of an index range  
"""
function sampling(range::Integer, perc::Float64; seed=42)
	randperm(MersenneTwister(seed), range)[1:ceil(Integer, perc * range)]
end

# ╔═╡ 00d20af2-7992-11eb-188e-55d5bf3fc1b3
begin
	## Since there are fewer risky loans than safe loans, find the ratio of the sizes
	## and use that percentage to undersample the safe loans.
	nsl = size(safe_loans_raw)[1]
	perc = size(risky_loans_raw)[1] / nsl

	risky_loans = risky_loans_raw
	ixes = sampling(nsl, perc; seed=42)
	safe_loans = safe_loans_raw[ixes, :]   ## down sample safe loans
	loans_data = [risky_loans; safe_loans] ## concatenate 
end

# ╔═╡ 1f4430b2-7999-11eb-1fd1-ddf50fb7a6dd
md"""
Now, let's verify that the resulting percentage of safe and risky loans are each nearly 50%.
"""

# ╔═╡ 1f29904a-7999-11eb-1007-4da27f7c6677
begin
	np_safe_loans, np_risky_loans = loan_ratios(safe_loans, risky_loans, n_loans=size(loans_data)[1])

	with_terminal() do
		@printf("Percentage of safe loans: %1.2f%%\n", np_safe_loans * 100.)
		@printf("Percentage of risky loans: %1.2f%%\n", np_risky_loans * 100.)
		@printf("Total number of loans in our new dataset: %d\n", size(loans_data)[1])
	end
end

# ╔═╡ 1f0ac002-7999-11eb-0bb5-63faa4dcb6cc
md"""
**Note**: There are many approaches for dealing with imbalanced data, including some where we modify the learning algorithm. These approaches are beyond the scope of this course. 

For this assignment, we use the simplest possible approach, where we subsample the overly represented class to get a more balanced dataset. In general, and especially when the data is highly imbalanced, we recommend using more advanced methods.
"""

# ╔═╡ f0a6c596-7999-11eb-0629-979335f89922
md"""
### Split data into training and validation sets

We split the data into training and validation sets using an 80/20 split and specifying `seed=1` so everyone gets the same results.

**Note**: In previous assignments, we have called this a **train-test split**. However, the portion of data that we don't train on will be used to help **select model parameters** (this is known as model selection). Thus, this portion of data should be called a **validation set**. Recall that examining performance of various potential models (i.e. models with different parameters) should be on validation set, while evaluation of the final selected model should always be on test data. 

Typically, we would also save a portion of the data (a real test set) to test our final model on or use cross-validation on the training set to select our final model. But for the learning purposes of this assignment, we won't do that.
"""

# ╔═╡ f0890d6e-7999-11eb-3482-4769d6489475
begin
	train_data, validation_data = train_test_split(loans_data; split=0.8, seed=42);
	size(train_data), size(validation_data)
end

# ╔═╡ 68e89cf0-79fb-11eb-24ff-d37b75bebe2e
first(train_data, 3)

# ╔═╡ f070b8f2-7999-11eb-2849-f56a3c74659b
md"""
### Use decision tree to build a classifier

Now, let's use the MLJ decision tree learner to create a loan prediction model on the training data. (In the next assignment, you will implement your own decision tree learning algorithm.) 

Our feature columns and target column have already been decided above.

"""

# ╔═╡ a7f44f2a-799c-11eb-34ff-9b3613ca036c
# Pkg.add("MLJDecisionTreeInterface")

# ╔═╡ 971c2ac2-799c-11eb-1d64-d7e7ff02159c
@load DecisionTreeClassifier pkg=DecisionTree

# ╔═╡ c9e3a334-79fa-11eb-39e1-c1565af8a588
function data_prep_Xy(df::DF, target; 
		features_st=Features_Scitype) where {DF <: Union{DataFrame, SubDataFrame}}
	y, X = unpack(df, 
		==(target),               # y is the `target` column
        !=(target);               # X is the rest, except the `target` column
        target => Multiclass{2},  # y is -1 or +1
        features_st...)
	return (X, y)
end

# ╔═╡ a4d03076-7a09-11eb-3f1b-e503cf36f511
function data_prep(df::DF, target; 
		features_st=Features_Scitype) where {DF <: Union{DataFrame, SubDataFrame}}
	coerce(df, features_st...)
end

# ╔═╡ 77751a08-79a0-11eb-2c16-f342efd58086
function dt_runner(df::DF, features, target; 
		max_depth=-1) where {DF <: Union{DataFrame, SubDataFrame}}	
	X, y = data_prep_Xy(df, target)
	
	# dt_clf = MLJDecisionTreeInterface.DecisionTreeClassifier(
	# 	max_depth = max_depth,
	# 	min_samples_leaf = 1,
	# 	min_samples_split = 2,
	# 	min_purity_increase = 0.0,
	# 	n_subfeatures = 0,
	# 	post_prune = false,
	# 	merge_purity_threshold = 1.0,
	# 	pdf_smoothing = 0.0,
	# 	display_depth = 5
	# )
	
	pipeclf = @pipeline(
	 	OneHotEncoder(),		
		MLJDecisionTreeInterface.DecisionTreeClassifier()
	)

	pipeclf.decision_tree_classifier.max_depth=max_depth
	
	dt_mach = machine(pipeclf, X, y)
	fit!(dt_mach, verbosity=0)
	dt_mach
end

# ╔═╡ 92fe3f40-798d-11eb-190c-81209cbb835b
function dt_runner_alt(df::DF, features, target; 
		max_depth=3) where {DF <: Union{DataFrame, SubDataFrame}}	
	X, y = data_prep_Xy(df, target)
	
	pipeclf = @pipeline(
		OneHotEncoder(),
		MLJDecisionTreeInterface.DecisionTreeClassifier()
	)
	
	r_mpi = range(pipeclf, :(pipeclf.decision_tree_classifier.max_depth), 
		lower=1, upper=max_depth)
	# r_msl = range(pipeclf, :(pipeclf.decision_tree_classifier.min_samples_leaf), 
	# 	lower=1, upper=50)
	
	tm = TunedModel(model=pipeclf, ranges=[r_mpi], tuning=Grid(resolution=8), 
		resampling=CV(nfolds=5, rng=112),
		operation=predict_mode, 
		measure=misclassification_rate, 
		check_measure=false)
	
	mtm = machine(tm, Xt, y)
	fit!(mtm, rows=1:size(df)[1], verbosity=1)

	mtm
end

# ╔═╡ 08fc6234-799e-11eb-072e-f31cc84a982a
begin
	dt_model = dt_runner(train_data, Features, target; max_depth=6)

	with_terminal() do
		println("Summary dt_model:")
		println("--------------------------------------")
		println(fitted_params(dt_model))
		println("--------------------------------------")
	end
end

# ╔═╡ f76d8430-79a7-11eb-1d64-d7e7ff02159c
md"""

Error / References:

  - https://github.com/alan-turing-institute/MLJ.jl/issues/423

  - https://stackoverflow.com/questions/66365311/the-scitype-of-x-in-machine-is-incompatible-with-model

  - https://alan-turing-institute.github.io/MLJ.jl/stable/common_mlj_workflows/#Data-ingestion-1

"""

# ╔═╡ dcc73932-79ac-11eb-196a-4d0b50beb96d
with_terminal() do
	println(report(dt_model))
end

# ╔═╡ d8d0090a-79a0-11eb-1d64-d7e7ff02159c
md"""
#### Building a smaller tree

A tree can be hard to visualize graphically, and moreover, it may overfit.. Here, we instead learn a smaller model with max depth of 2 to gain some intuition and to understand the learned tree more.
"""

# ╔═╡ 6f5b75f0-7a03-11eb-1e21-ab19398446c2
begin
	dt_small_model = dt_runner(train_data, Features, target; max_depth=2)

	with_terminal() do
		println("Summary dt_small_model:")
		println("--------------------------------------")
		println(fitted_params(dt_small_model))
		println("--------------------------------------")
	end
end

# ╔═╡ 0bd3a464-7a18-11eb-2e7d-c7246d8cc30d
md"""
#### Building a bigger tree
"""

# ╔═╡ 1f101802-7a18-11eb-3e07-e5f88b6f9b58
begin
	dt_big_model = dt_runner(train_data, Features, target; max_depth=10)

	with_terminal() do
		println("Summary dt_big_model:")
		println("--------------------------------------")
		println(fitted_params(dt_big_model))
		println("--------------------------------------")
	end
end

# ╔═╡ 6f3bc75a-7a03-11eb-1df3-392d8e3e0363
md"""
#### Making predictions

Let's consider two positive and two negative examples from the validation set and see what the model predicts. We will do the following:

  - Predict whether or not a loan is safe.
  - Predict the probability that a loan is safe.


"""

# ╔═╡ 6f2014a6-7a03-11eb-1ec1-abd58d12aac7
begin
	n_validation_data = data_prep(validation_data, target)
	
	valid_safe_loans = filter(target => ==(1), n_validation_data) 	  
	valid_risky_loans = filter(target => ==(-1), n_validation_data)

	sample_valid_data = vcat(valid_safe_loans[1:2, :], valid_risky_loans[1:2, :])
	sample_valid_data
end

# ╔═╡ c6f5a1ea-7a0b-11eb-0dd9-f79fc57abf0e
begin
	using Distributions
	
	y = sample_valid_data.safe_loans
	ŷ = predict(dt_model, select(sample_valid_data, Features))
	
	L = levels(y)
	broadcast(pdf, ŷ, y), pdf(ŷ, L)
end

# ╔═╡ 3882268e-7a06-11eb-2cfa-8bdfdb75dc5a
#X_sample_valid_data, y_sample_valid_data = data_prep(sample_valid_data, target; 
#		features_st=Features_Scitype);

# ╔═╡ a0f6600c-7a08-11eb-0190-a1eea0f3ef62
function get_levels(col)
	length(levels(col)), levels(col)
end

# ╔═╡ 3ff06776-7a08-11eb-0187-4191b4637cbe
# length(levels(train_data.grade)), levels(train_data.grade)
get_levels(train_data.grade)

# ╔═╡ 747f7110-7a08-11eb-16d1-6333aa1c0df5
get_levels(train_data.sub_grade)

# ╔═╡ 74601010-7a08-11eb-164b-8de382485329
get_levels(sample_valid_data.grade)

# ╔═╡ 74445e4c-7a08-11eb-067f-ffc2701a5690
# levels!(sample_valid_data.grade, levels(train_data.grade))

# ╔═╡ 742ee4d6-7a08-11eb-21c8-0d51ac69554a


# ╔═╡ b39d761c-7a05-11eb-2f0f-27f32329cb2f
md"""
#### Explore label predictions

Now, we will use our model  to predict whether or not a loan is likely to default. For each row in the `sample_validation_data`, use the `dt_model` to predict whether or not the loan is classified as a **safe loan**. 
"""

# ╔═╡ b3833b4e-7a05-11eb-09f6-43a205481b3b
begin
	y_hat_mode = predict_mode(dt_model, select(sample_valid_data, Features))
	
	perc_correct_preds = sum(y_hat_mode .== sample_valid_data.safe_loans) / size(sample_valid_data)[1]
	(percentage_prediction=perc_correct_preds,)
end

# ╔═╡ b368f158-7a05-11eb-17e9-8ddaea0b2386
md"""
**Quiz Question:** What percentage of the predictions on `sample_valid_data` did `dt_model` get correct?
  - cf. above cell
"""

# ╔═╡ d927b594-7a0c-11eb-3329-11481535ab85
begin
	ix = argmax(broadcast(pdf, ŷ, y))
	ŷ_prob = broadcast(pdf, ŷ, y)
	(ix=ix, value=ŷ_prob[ix])
end

# ╔═╡ b34f5fb8-7a05-11eb-1399-d9cd7c8ded35
md"""
**Quiz Question**: Which loan has the highest probability of being classified as a safe loan?
  - in this case the third (index 3, *Julia indexes start from 1*)

"""

# ╔═╡ 1194ab16-7a15-11eb-127e-1dec3928fc2a
md"""
**Checkpoint:** Can you verify that for all the predictions with `probability >= 0.5`, the model predicted the label **+1**?
"""

# ╔═╡ b8c7a3c0-7a1d-11eb-296e-ef4d312ffcfa
ŷ_prob, y

# ╔═╡ c89eeb2e-7a1c-11eb-2278-ab37447cf22d
begin
	# indexes for which y is 1
	ixes_ = map(t -> t[1], 
		filter(((ix, v)=t) -> v, collect(enumerate(y .==  1))))

	# all probs matching y .== 1 are ≥ 0.5
	@test all(ŷ_prob[ixes_, :] .≥ 0.5)
end

# ╔═╡ 47718414-7a1f-11eb-0b31-6d187e952bad
md"""
##### Tricky predictions!

Now, we will explore something pretty interesting. For each row in the `sample_valid_data`, what is the probability (according to dt_small_model) of a loan being classified as safe?

"""

# ╔═╡ c850fd10-7a1c-11eb-3b93-1bbeb7098830
begin
	# ŷ_vec = predict_mode(dt_small_model, select(sample_valid_data, Features))
	ŷ_dt_small_ = predict(dt_small_model, select(sample_valid_data, Features))
	
	# ŷ_vec, 
	broadcast(pdf, ŷ_dt_small_, y)
end

# ╔═╡ 12909320-7a15-11eb-13df-691a76b3a823
md"""
### Evaluating accuracy of the decision tree model

Recall that the accuracy is defined as follows:

```math
\mbox{accuracy} = \frac{\mbox{\# correctly classified examples}}{\mbox{\# total examples}}
```

Let us start by evaluating the accuracy of the `dt_small_model` and `dt_model` on the training data
"""

# ╔═╡ 924613f6-7a15-11eb-05a4-8351feaf855c


# ╔═╡ 931f5666-7a15-11eb-0042-cd16bc34e020
function accuracy(ŷ, y)
	sum(ŷ .== y) / size(y)[1] 
end
	

# ╔═╡ b3352738-7a05-11eb-13eb-f341e0e2e6df
begin
	ys = n_validation_data.safe_loans
	
	ŷ_dt = predict_mode(dt_model, select(n_validation_data, Features))
	ŷ_dt_small = predict_mode(dt_small_model, select(n_validation_data, Features))
	ŷ_dt_big = predict_mode(dt_big_model, select(n_validation_data, Features))
	
	# mode.(y_hat)
	# broadcast(pdf, y_hats, ys, ) #[1:2]
	acc_dt = accuracy(ŷ_dt, ys)
	acc_dt_small = accuracy(ŷ_dt_small, ys)
	acc_dt_big = accuracy(ŷ_dt_big, ys)
	
	(acc_dt=acc_dt |> a_ -> round(a_; digits=3), 
		acc_dt_small=acc_dt_small |> a_ -> round(a_; digits=3),
		acc_dt_big=acc_dt_big |> a_ -> round(a_; digits=3))
end

# ╔═╡ 4dce4096-7a19-11eb-0bd8-f14ec24ed701
md"""
### Quantifying the cost of mistakes

Every mistake the model makes costs money. In this section, we will try and quantify the cost of each mistake made by the model.

Assume the following:

* **False negatives**: Loans that were actually safe but were predicted to be risky. This results in an oppurtunity cost of losing a loan that would have otherwise been accepted. 
* **False positives**: Loans that were actually risky but were predicted to be safe. These are much more expensive because it results in a risky loan being given. 
* **Correct predictions**: All correct predictions don't typically incur any cost.


Let's write code that can compute the cost of mistakes made by the model. Complete the following 4 steps:
1. First, let us compute the predictions made by the model.
1. Second, compute the number of false positives.
2. Third, compute the number of false negatives.
3. Finally, compute the cost of mistakes made by the model by adding up the costs of true positives and false positives.

"""

# ╔═╡ fdd913ca-7a1a-11eb-3bee-6f04974e3141
function calc_fp(ŷ, y)
	filter(((ŷ_, y_)=t) -> ŷ_ == y_ == 1, 
		collect(zip(ŷ .== 1, y .== -1))) |> length
end

# ╔═╡ c111df1c-7a1a-11eb-1474-b96dcad21315
function calc_fn(ŷ, y)
	filter(((ŷ_, y_)=t) -> ŷ_ == y_ == 1, 
		collect(zip(ŷ .== -1, y .== 1))) |> length
end

# ╔═╡ 66bc0b4c-7a19-11eb-1d94-df9fa1b7e7b5
md"""
False positives are predictions where the model predicts +1 but the true label is -1. Complete the following code block for the number of false positives:
"""

# ╔═╡ 7120d40c-7a1a-11eb-33f2-8f6b79fcb83a
fp_dt, fp_dt_big = calc_fp(ŷ_dt, ys),  calc_fp(ŷ_dt_big, ys)

# ╔═╡ 6d3396cc-7a19-11eb-0c3f-3d0456bff8c5
md"""
False negatives are predictions where the model predicts -1 but the true label is +1. Complete the following code block for the number of false negatives:
"""

# ╔═╡ 6d176dec-7a19-11eb-25ae-09cf985eb22a
fn_dt, fn_dt_big = calc_fn(ŷ_dt, ys), calc_fn(ŷ_dt_big, ys)

# ╔═╡ 6cf9fa46-7a19-11eb-249a-d300a87184cf
md"""
**Quiz Question:** Let us assume that each mistake costs money:
  - Assume a cost of \$10,000 per false negative.
  - Assume a cost of \$20,000 per false positive.

What is the total cost of mistakes made by `decision_tree_model` on `validation_data`?
"""

# ╔═╡ 8b58e726-7a1a-11eb-076b-e3b226df3fc2
(dt_model_cost=fp_dt * 20000 + fn_dt * 10000, dt_big_model_cost=fp_dt_big * 20000 + fn_dt_big * 10000)

# ╔═╡ Cell order:
# ╟─51dfdce2-7980-11eb-229a-3b3196ded703
# ╠═90bdce42-7980-11eb-08ad-0f39e65e1864
# ╟─a36cad10-7980-11eb-212f-67774be5fd35
# ╠═a36b25ee-7980-11eb-2308-e1b576d9883e
# ╟─a36b0642-7980-11eb-3a6f-e3f48bc58616
# ╠═19b49852-7981-11eb-1d65-99a8f9119422
# ╠═882c144c-7989-11eb-0e5d-4d2b0d13b459
# ╠═e84e8f1c-798e-11eb-23e6-0dafc1deaff4
# ╠═d5331906-798f-11eb-21c5-bf17a158c639
# ╟─87ede65a-798d-11eb-2610-7db61052073b
# ╠═9405f590-798d-11eb-3a55-619e7d2c5cb6
# ╠═93e75248-798d-11eb-2e4b-cb8072c75df7
# ╟─93ce629a-798d-11eb-02fe-7b0556848660
# ╟─93b3c4aa-798d-11eb-04e9-f526070ac3eb
# ╠═939cde84-798d-11eb-244a-9ba95ee1ada7
# ╟─937f41c6-798d-11eb-0eec-ffd7cb58b2b6
# ╠═9368145e-798d-11eb-2232-dd7572dfb755
# ╠═9350b5f4-798d-11eb-2c9f-a1cbd7526d1a
# ╟─93194734-798d-11eb-3d96-1f8a46085885
# ╟─0120efaa-7992-11eb-313e-6b5e4dfd4800
# ╠═1671f0ac-7992-11eb-20ff-912bf4353a24
# ╟─16564a14-7992-11eb-3600-cd4cf1573933
# ╟─0aa86afe-79ff-11eb-0d39-4fee0ee8c7fc
# ╠═34c2bb96-79aa-11eb-1d64-d7e7ff02159c
# ╠═67b8b802-79fa-11eb-241a-17e808148f7c
# ╟─e9c1bcc8-79fe-11eb-1437-9530a1eefbdb
# ╠═1d6247c6-7a01-11eb-39e1-c1565af8a588
# ╟─f2c7be06-7992-11eb-2c95-979c0965fe77
# ╠═f29fef50-7992-11eb-2ed8-2bf706c9828a
# ╟─1629b576-7992-11eb-1fef-a3538166d17d
# ╠═16055156-7992-11eb-3461-17b098dc3a27
# ╟─00f5c508-7992-11eb-1087-e7c63d454e06
# ╠═65959ca2-7996-11eb-2582-5f6f479ff6ff
# ╠═b3fef5fa-7996-11eb-0d80-45603e97e1bb
# ╠═b9cbc562-7996-11eb-1fb4-49c80af4690d
# ╠═00d20af2-7992-11eb-188e-55d5bf3fc1b3
# ╟─1f4430b2-7999-11eb-1fd1-ddf50fb7a6dd
# ╠═1f29904a-7999-11eb-1007-4da27f7c6677
# ╟─1f0ac002-7999-11eb-0bb5-63faa4dcb6cc
# ╟─f0a6c596-7999-11eb-0629-979335f89922
# ╠═20f35fc2-799a-11eb-12e9-4588c897199d
# ╠═f0890d6e-7999-11eb-3482-4769d6489475
# ╠═68e89cf0-79fb-11eb-24ff-d37b75bebe2e
# ╟─f070b8f2-7999-11eb-2849-f56a3c74659b
# ╠═a7f44f2a-799c-11eb-34ff-9b3613ca036c
# ╠═971c2ac2-799c-11eb-1d64-d7e7ff02159c
# ╠═c9e3a334-79fa-11eb-39e1-c1565af8a588
# ╠═a4d03076-7a09-11eb-3f1b-e503cf36f511
# ╠═77751a08-79a0-11eb-2c16-f342efd58086
# ╠═92fe3f40-798d-11eb-190c-81209cbb835b
# ╠═08fc6234-799e-11eb-072e-f31cc84a982a
# ╟─f76d8430-79a7-11eb-1d64-d7e7ff02159c
# ╠═dcc73932-79ac-11eb-196a-4d0b50beb96d
# ╟─d8d0090a-79a0-11eb-1d64-d7e7ff02159c
# ╠═6f5b75f0-7a03-11eb-1e21-ab19398446c2
# ╟─0bd3a464-7a18-11eb-2e7d-c7246d8cc30d
# ╠═1f101802-7a18-11eb-3e07-e5f88b6f9b58
# ╟─6f3bc75a-7a03-11eb-1df3-392d8e3e0363
# ╠═6f2014a6-7a03-11eb-1ec1-abd58d12aac7
# ╠═3882268e-7a06-11eb-2cfa-8bdfdb75dc5a
# ╠═a0f6600c-7a08-11eb-0190-a1eea0f3ef62
# ╠═3ff06776-7a08-11eb-0187-4191b4637cbe
# ╠═747f7110-7a08-11eb-16d1-6333aa1c0df5
# ╠═74601010-7a08-11eb-164b-8de382485329
# ╠═74445e4c-7a08-11eb-067f-ffc2701a5690
# ╠═742ee4d6-7a08-11eb-21c8-0d51ac69554a
# ╟─b39d761c-7a05-11eb-2f0f-27f32329cb2f
# ╠═b3833b4e-7a05-11eb-09f6-43a205481b3b
# ╟─b368f158-7a05-11eb-17e9-8ddaea0b2386
# ╠═c6f5a1ea-7a0b-11eb-0dd9-f79fc57abf0e
# ╠═d927b594-7a0c-11eb-3329-11481535ab85
# ╟─b34f5fb8-7a05-11eb-1399-d9cd7c8ded35
# ╟─1194ab16-7a15-11eb-127e-1dec3928fc2a
# ╠═b8c7a3c0-7a1d-11eb-296e-ef4d312ffcfa
# ╠═c89eeb2e-7a1c-11eb-2278-ab37447cf22d
# ╟─47718414-7a1f-11eb-0b31-6d187e952bad
# ╠═c850fd10-7a1c-11eb-3b93-1bbeb7098830
# ╟─12909320-7a15-11eb-13df-691a76b3a823
# ╠═924613f6-7a15-11eb-05a4-8351feaf855c
# ╠═931f5666-7a15-11eb-0042-cd16bc34e020
# ╠═b3352738-7a05-11eb-13eb-f341e0e2e6df
# ╟─4dce4096-7a19-11eb-0bd8-f14ec24ed701
# ╠═fdd913ca-7a1a-11eb-3bee-6f04974e3141
# ╠═c111df1c-7a1a-11eb-1474-b96dcad21315
# ╟─66bc0b4c-7a19-11eb-1d94-df9fa1b7e7b5
# ╠═7120d40c-7a1a-11eb-33f2-8f6b79fcb83a
# ╟─6d3396cc-7a19-11eb-0c3f-3d0456bff8c5
# ╠═6d176dec-7a19-11eb-25ae-09cf985eb22a
# ╟─6cf9fa46-7a19-11eb-249a-d300a87184cf
# ╠═8b58e726-7a1a-11eb-076b-e3b226df3fc2
