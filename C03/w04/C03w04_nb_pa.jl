### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 77311686-7a2b-11eb-2bc7-95b2a2211969
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

# ╔═╡ 0d124834-7ef3-11eb-1d64-d7e7ff02159c
begin
	const DF = AbstractDataFrame # Union{DataFrame, SubDataFrame}
	
	include("./utils.jl");	
	include("./dt_utils.jl");
end

# ╔═╡ 5435ff66-7a2b-11eb-1474-b96dcad21315
md"""
## C03w04: Decision Trees in Practice

In this assignment we will explore various techniques for preventing overfitting in decision trees. We will extend the implementation of the binary decision trees that we implemented in the previous assignment. <br/>
We will have use our solutions from the previous assignment and extend them.

In this assignment we will:
  -  Implement binary decision trees with different early stopping methods.
  -  Compare models with different stopping parameters.
  -  Visualize the concept of overfitting in decision trees.
"""

# ╔═╡ 771130dc-7a2b-11eb-170f-934c11fceede
md"""
#### Load LendingClub dataset
"""

# ╔═╡ 76f62e40-7a2b-11eb-095e-f36e207f06a2
begin
	loans =  CSV.File(("../../ML_UW_Spec/C03/data/lending-club-data.csv"); 
	header=true) |> DataFrame;
	size(loans)
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
Unlike the previous assignment where we used several features, in this assignment, we will just be using 4 categorical
features: 

  1. grade of the loan 
  2. the length of the loan term
  3. the home ownership status: own, mortgage, rent
  4. number of years of employment.

Since we are building a binary decision tree, we will have to convert these categorical features to a binary representation in a subsequent section using 1-hot encoding.
"""

# ╔═╡ 768d6b08-7a2b-11eb-388a-8763c8da0a51
begin
  	const Features = [
		:grade,          # grade of the loan            
		:emp_length,     # number of years of employment
		:home_ownership, # home_ownership status: own, mortgage or rent
		:term,           # the term of the loan
	]

	const Target = :safe_loans # prediction target (y) (+1 means safe, -1 is risky)

	# Extract the feature columns and target column
	select!(loans, [Features..., Target]);
	length(names(loans)), names(loans)
end

# ╔═╡ 7673d8e4-7a2b-11eb-02b4-2f837181b4e9
md"""
#### Subsample dataset to make sure classes are balanced

Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points.
"""

# ╔═╡ 1f459740-7ef3-11eb-34ff-9b3613ca036c
begin
	len_loans₀	 = size(loans)[1]	
  	safe_loans₀, risky_loans₀ = df_partition(loans, Target)
  	p_safe_loans₀, p_risky_loans₀ = df_ratios(safe_loans₀, risky_loans₀, len_loans₀);

  	with_terminal() do
    	@printf("Number of safe loans  : %d\n", size(safe_loans₀)[1])
    	@printf("Number of risky loans : %d\n", size(risky_loans₀)[1])
    	@printf("Percentage of safe loans:  %1.2f%%\n", p_safe_loans₀ * 100.)
    	@printf("Percentage of risky loans: %3.2f%%\n", p_risky_loans₀ * 100.)
  	end
end

# ╔═╡ 32a93648-7ef3-11eb-071c-0fb7dd06e885
loans_data = resample!(safe_loans₀, risky_loans₀);

# ╔═╡ 31f2811e-7ef3-11eb-2890-7577dda5d4de
begin
	len_loans₁	 = size(loans_data)[1]
  	safe_loans₁, risky_loans₁ = df_partition(loans_data, Target)
  	p_safe_loans₁, p_risky_loans₁ = df_ratios(safe_loans₁, risky_loans₁, len_loans₁);

 	with_terminal() do
    	@printf("Number of safe loans  : %d\n", size(safe_loans₁)[1])
    	@printf("Number of risky loans : %d\n", size(risky_loans₁)[1])
    	@printf("Percentage of safe loans:  %1.2f%%\n", p_safe_loans₁ * 100.)
    	@printf("Percentage of risky loans: %3.2f%%\n", p_risky_loans₁ * 100.)
  	end
end

# ╔═╡ 760c171a-7a2b-11eb-1341-19ba5a1824a1
md"""
#### Transform categorical data into binary features 

In this assignment, we will implement **binary decision trees** (decision trees for binary features, a specific case of categorical variables taking on two values, e.g., true/false). Since all of our features are currently categorical features, we want to turn them into binary features. 

For instance, the `home_ownership` feature represents the home ownership status of the loanee, which is either `own`, `mortgage` or `rent`. For example, if a data point has the feature 
```
   {'home_ownership': 'RENT'}
```
we want to turn this into three features: 
```
 { 
   'home_ownership = OWN'      : 0, 
   'home_ownership = MORTGAGE' : 0, 
   'home_ownership = RENT'     : 1
 }
```
"""

# ╔═╡ 75d98df2-7a2b-11eb-296e-ef4d312ffcfa
loans_data[1:5, :]

# ╔═╡ 75c1a64e-7a2b-11eb-21f6-55dd3a8ce164
begin
	hot_encode!(loans_data)
	loans_data[1:5, :]
end

# ╔═╡ 75a52c44-7a2b-11eb-2f02-2312f3a3dac9
begin
	features = names(loans_data) |> a_ -> Symbol.(a_);
	deleteat!(features, 
		findall(x -> x == Target, features)) 
end

# ╔═╡ 75736a9c-7a2b-11eb-2278-ab37447cf22d
## Expect 25 features after hot-encoding
@test length(features) == 25

# ╔═╡ 116eb5f8-7a33-11eb-2c4c-5f937540f987
length(loans_data.grade_A)
# vs 46508 in original notebook

# ╔═╡ a5c56d68-7a32-11eb-2ef6-e58ed65dc561
md"""
**Checkpoint:** Make sure the following answers match up.
"""

# ╔═╡ a5a1cdf6-7a32-11eb-18dd-0dfee8eb885c
@test sum(loans_data.grade_A) == 6537
# vs 6422 in original notebook

# ╔═╡ a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
md"""
#### Train-test split

We split the data into a train test split with 80% of the data in the training set and 20% of the data in the test set. 
"""

# ╔═╡ 6fad8afe-7a33-11eb-3208-6b289c3ccd23
begin
	train_data, validation_data = train_test_split(loans_data; split=0.8, seed=42);
	size(train_data), size(validation_data)
    ## vs ((37224, 26), (9284, 26)) ...
end

# ╔═╡ 6f88e4f6-7a33-11eb-10cf-e3e8608e616b
md"""
### Early stopping methods for decision trees

In this section, we will extend the **binary tree implementation** from the previous assignment in order to handle some early stopping conditions. Recall the 3 early stopping methods that were discussed in lecture:

  1. Reached a **maximum depth**. (set by parameter `max_depth`).
  2. Reached a **minimum node size**. (set by parameter `min_node_size`).
  3. Don't split if the **gain in error reduction** is too small. (set by parameter `min_error_reduction`).

For the rest of this assignment, we will refer to these three as **early stopping conditions 1, 2, and 3**.

#### Early stopping condition 1: Maximum depth

Recall that we already implemented the maximum depth stopping condition in the previous assignment. In this assignment, we will experiment with this condition a bit more and also write code to implement the 2nd and 3rd early stopping conditions.

We will be reusing code from the previous assignment and then building upon this. 
"""

# ╔═╡ 2e859492-7b2b-11eb-1d64-d7e7ff02159c
md"""
#### Early stopping condition 2: Minimum node size

The function `reached_minimum_node_size` takes 2 arguments:

1. The `data` (from a node)
2. The minimum number of data points that a node is allowed to split on, `min_node_size`.

This function simply calculates whether the number of data points at a given node is less than or equal to the specified minimum node size. This function will be used to detect this early stopping condition in the `decision_tree_create` function.
"""

# ╔═╡ 49f07e5c-7b2b-11eb-34ff-9b3613ca036c
function reached_minimum_node_size(data::DF, min_node_size) 
    """true when number of data points is ≤ to the minimum node size."""
    size(data)[1] ≤ min_node_size
end

# ╔═╡ 84eab9ca-7b2b-11eb-2890-7577dda5d4de
md"""
**Quiz Question:** Given an intermediate node with 6 safe loans and 3 risky loans, if the `min_node_size` parameter is 10, what should the tree learning algorithm do next?
  - it should stop (early), creating a leaf and return it
"""

# ╔═╡ 94fccdee-7b2b-11eb-071c-0fb7dd06e885
md"""
#### Early stopping condition 3: Minimum gain in error reduction

The function `error_reduction` takes 2 arguments:

  1. The error **before** a split, `error_before_split`.
  2. The error **after** a split, `error_after_split`.

This function computes the gain in error reduction, i.e., the difference between the error before the split and that after the split. This function will be used to detect this early stopping condition in the `decision_tree_create` function.
"""

# ╔═╡ a58dfb04-7b2b-11eb-30dd-677a45cd9920
function error_reduction(error_before_split, error_after_split)
	"""Return the error before the split minus the error after the split."""
	error_before_split - error_after_split
end

# ╔═╡ c4c23a3c-7b2b-11eb-37bc-7df8411fb645
md"""
**Quiz Question:** Assume an intermediate node has 6 safe loans and 3 risky loans.  For each of 4 possible features to split on, the error reduction is 0.0, 0.05, 0.1, and 0.14, respectively. If the **minimum gain in error reduction** parameter is set to 0.2, what should the tree learning algorithm do next?
  - it should stop (early), creating a leaf and return it
"""

# ╔═╡ ce3be5b8-7b2b-11eb-1bbb-e39f6ba35768
md"""
#### Grabbing binary decision tree helper functions from past assignment


Recall from the previous assignment that we wrote a function `inter_node_num_mistakes` that calculates the number of **misclassified examples** when predicting the **majority class**. This is used to help determine which feature is best to split on at a given node of the tree.

cf. include
"""

# ╔═╡ 6f3011da-7a33-11eb-2cd0-b7bb04b7b545
## TODO move...

with_terminal() do
	@testset "begin sanity checks" begin
		## Test case 1
		ex1_labels = DataFrame(:x => [-1, -1, 1, 1, 1])
		@test inter_node_num_mistakes(ex1_labels.x) == 2

		## Test case 2
		ex2_labels = [-1, -1, 1, 1, -1, 1, 1]
		@test inter_node_num_mistakes(ex2_labels) == 3

		## Test case 3
		ex3_labels = DataFrame(:x => [-1, -1, -1, -1, 1, -1, 1])
		@test inter_node_num_mistakes(ex3_labels.x) == 2
	end
end

# ╔═╡ ce1d2376-7b2b-11eb-14c6-3dd9dd0d2303
md"""
We then wrote a function `best_splitting_feature` that finds the best feature to split on given the data and a list of features to consider.

cf. include
"""

# ╔═╡ ce025442-7b2b-11eb-31b8-69bdfe81ed69
@test best_splitting_feature(train_data, features, Target) == Symbol("term_ 60 months")

# ╔═╡ cde76c36-7b2b-11eb-2ad1-017f6ac940dd
md"""
Finally, recall the function create_leaf from the previous assignment, which creates a leaf node given a set of target values.

cf. include
"""

# ╔═╡ cdcdcc7c-7b2b-11eb-3b98-bd0e15b4de11
md"""
### Incorporating new early stopping conditions in binary decision tree implementation


Now, you will implement a function that builds a decision tree handling the three early stopping conditions described in this assignment.  In particular, you will write code to detect early stopping conditions 2 and 3.  You implemented above the functions needed to detect these conditions.  The 1st early stopping condition, **max_depth**, was implemented in the previous assigment and you will not need to reimplement this.  In addition to these early stopping conditions, the typical stopping conditions of having no mistakes or no more features to split on (which we denote by "stopping conditions" 1 and 2) are also included as in the previous assignment.

**Implementing early stopping condition 2: minimum node size:**

* **Step 1:** Use the function **reached_minimum_node_size** that you implemented earlier to write an if condition to detect whether we have hit the base case, i.e., the node does not have enough data points and should be turned into a leaf. Don't forget to use the `min_node_size` argument.
* **Step 2:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.


**Implementing early stopping condition 3: minimum error reduction:**

**Note:** This has to come after finding the best splitting feature so we can calculate the error after splitting in order to calculate the error reduction.

* **Step 1:** Calculate the **classification error before splitting**.  Recall that classification error is defined as:

$$\text{classification error} = \frac{\text{\# mistakes}}{\text{\# total examples}}$$


* **Step 2:** Calculate the **classification error after splitting**. This requires calculating the number of mistakes in the left and right splits, and then dividing by the total number of examples.
* **Step 3:** Use the function **error_reduction** to that you implemented earlier to write an if condition to detect whether  the reduction in error is less than the constant provided (`min_error_reduction`). Don't forget to use that argument.
* **Step 4:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.

"""

# ╔═╡ b6fc15ae-7a37-11eb-071c-0fb7dd06e885
function decision_tree_create(df::DF, features, target;
		curr_depth=0, max_depth=10, min_node_size=1, min_error_reduction=0.0,
		verbose=false) where {DF <: Union{DataFrame, SubDataFrame}}
	
    rem_features = copy(features) # Make a copy of the features.
    
    target_vals = df[!, target]
    if verbose
      println("-------------------------------------------------------------")
      @printf("Subtree, depth = %s (%s data points).\n", curr_depth, 
			size(target_vals)[1])
	end

    if inter_node_num_mistakes(target_vals) == 0
		## Stopping cond1: no mistakes at current node?
        verbose && @printf("Stopping condition 1 reached.")
        return create_leaf(target_vals)
			
	elseif length(rem_features) == 0
    	## Stopping cond2: no remaining features to consider splitting on?
        verbose && @printf("Stopping condition 2 reached.")
        return create_leaf(target_vals)
			
	elseif curr_depth ≥ max_depth
    	## Additional stopping condition (limit tree depth)
        verbose && @printf("Early stopping condition 1 reached. Maximum depth.")
        return create_leaf(target_vals)
		
	elseif reached_minimum_node_size(df, min_node_size)
		## Early stopping condition 2: 
        verbose && @printf("Early stopping condition 2 reached. Minimum node size.")
        return create_leaf(target_vals) 
	end

    ## Now Find the best splitting feature 
    splitting_feature = best_splitting_feature(df, features, target)
    
    # Split on the best feature that we found. 
    l_split = df[df[!, splitting_feature] .== 0, :]
    r_split = df[df[!, splitting_feature] .== 1, :]
	
	## Early stopping condition 3: Minimum error reduction (prep)
	err_bef_split = inter_node_num_mistakes(target_vals) / size(df)[1]
	
	## Calculate the error after splitting 
    l_mistakes = inter_node_num_mistakes(l_split[!, target]) 
    r_mistakes = inter_node_num_mistakes(r_split[!, target])
    err_aft_split = (l_mistakes + r_mistakes) / size(df)[1]
	
	if error_reduction(err_bef_split, err_aft_split) < min_error_reduction
        verbose && @printf("Early stopping condition 3. Minimum error reduction.")
        return create_leaf(target_vals)
	end
	
	deleteat!(rem_features, 
		findall(x -> x == splitting_feature, rem_features))
    
    verbose && @printf("Split on feature %s. (%s, %s)\n", splitting_feature, 
		size(l_split)[1], size(r_split)[1])
    
    # Create a leaf node if the split is "perfect"
    if size(l_split)[1] == size(df)[1]
        verbose && println("Creating leaf node.")
        return create_leaf(l_split[!, target])
	end
	
    if size(r_split)[1] == size(df)[1]
        verbose && println("Creating leaf node.")
        return create_leaf(r_split[!, target])
	end

    # Recurse on left and right subtrees
    ltree = decision_tree_create(l_split, rem_features, target;
						curr_depth=curr_depth + 1, max_depth, 
						min_node_size, min_error_reduction, verbose)
    rtree = decision_tree_create(r_split, rem_features, target;
						curr_depth=curr_depth + 1, max_depth,
						min_node_size, min_error_reduction, verbose)

    return create_leaf(target_vals; 
						splitting_feature=string(splitting_feature), 
						left=ltree, 
						right=rtree, 
						is_leaf=false)			
end

# ╔═╡ cdb69d9c-7b2b-11eb-12fa-4d42c67876b2
md"""

Run the following test code to check the implementation. 
"""

# ╔═╡ cd9e2b4a-7b2b-11eb-20d9-87257af20d51
begin
	small_data_decision_tree = decision_tree_create(train_data, features, :safe_loans;
		max_depth=2, min_node_size=10, min_error_reduction=0.0, verbose=true)

	@test count_nodes(small_data_decision_tree) == 7
end

# ╔═╡ 753b3df2-7a2b-11eb-28c0-b5bdffef414b
md"""
### Build the tree!

Let's now train a tree model **ignoring early stopping conditions 2 and 3** so that we get the same tree as in the previous assignment. 

To ignore these conditions, we set `min_node_size=0` and `min_error_reduction=-1` (a negative value).

Call this tree `dt6_model`.

"""

# ╔═╡ be0730f6-7a3d-11eb-30dd-677a45cd9920
## first no early stopping for later comparison
dt6_model_nes = decision_tree_create(train_data, features, :safe_loans;
	max_depth=6, min_node_size=0, min_error_reduction=-1.)

# ╔═╡ a92a0af0-7b34-11eb-2890-7577dda5d4de
md"""
Now that your code is working, we will train a tree model on the **train_data** with
  - `max_depth = 6`
  - `min_node_size = 100`, 
  - `min_error_reduction = 0.0`
"""

# ╔═╡ c88e3ac2-7b34-11eb-0a69-8fbbd0859786
dt6_model = decision_tree_create(train_data, features, :safe_loans;
	max_depth=6, min_node_size=100, min_error_reduction=0.0)

# ╔═╡ c872733c-7b34-11eb-05df-8b16a1439d68
md"""
#### Making Predictions
"""

# ╔═╡ c85ab936-7b34-11eb-37bc-7df8411fb645
md"""
Recall that in the previous assignment you implemented a function `classify` to classify a new point `x` using a given `tree`.

cf. include


Now, let's consider the first example of the validation set and see what the dt6_model model predicts for this data point.
"""

# ╔═╡ c8405834-7b34-11eb-30dd-677a45cd9920
first_dp = validation_data[1, :]

# ╔═╡ 594b159e-7b35-11eb-071c-0fb7dd06e885
with_terminal() do
	println("Predicted class: $(classify(dt6_model, first_dp))")
end

# ╔═╡ 59304232-7b35-11eb-2890-7577dda5d4de
with_terminal() do
	classify(dt6_model, first_dp, annotate=true)
end

# ╔═╡ 59141350-7b35-11eb-34ff-9b3613ca036c
md"""
Let's now recall the prediction path for the decision tree learned in the previous assignment, which we recreated here as dt6_model_nes
"""

# ╔═╡ aca4b40c-7b35-11eb-20d9-87257af20d51
with_terminal() do
	classify(dt6_model_nes, first_dp, annotate=true)
end

# ╔═╡ ac83767a-7b35-11eb-0a51-851d7325ad47
md"""
**Quiz Question:** For `dt6_model` trained with `max_depth=6`, `min_node_size=100`, `min_error_reduction=0.0`, is the prediction path for `validation_set[0]` shorter, longer, or the same as for `dt6_model_nes` that ignored the early stopping conditions 2 and 3?
  - it is the same

**Quiz Question:** For `dt6_model` trained with `max_depth=6`, `min_node_size=100`, `min_error_reduction=0.0`, is the prediction path for **any point** always shorter, always longer, always the same, shorter or the same, or longer or the same as for `dt6_model_nes` that ignored the early stopping conditions 2 and 3?
  - It should be shorter or the same


**Quiz Question:** For a tree trained on **any** dataset using `max_depth=6`, `min_node_size=100`, `min_error_reduction=0.0`, what is the maximum number of splits encountered while making a single prediction?
  - max-depth, hence 6 
"""

# ╔═╡ ac686542-7b35-11eb-08e3-635d3737c018
md"""
#### Evaluating the model

Now let us evaluate the model that we have trained. You implemented this evaluation in the function `eval_classification_error` from the previous assignment.

Now, let's use this function to evaluate the classification error of `dt6_model` on the **validation_data**.
"""

# ╔═╡ ac4e3b42-7b35-11eb-0a69-8fbbd0859786
with_terminal() do
	@printf("%1.4f\n", 
		round(eval_classification_error(dt6_model, validation_data, Target), 
			digits=4))
end

# ╔═╡ c825a994-7b34-11eb-071c-0fb7dd06e885
md"""
Now, evaluate the validation error using `dt6_model_nes`.
"""

# ╔═╡ 3bf28f8e-7b37-11eb-2890-7577dda5d4de
with_terminal() do
	@printf("%1.4f\n", 
		round(eval_classification_error(dt6_model_nes, validation_data, Target), 
			digits=4))
end

# ╔═╡ 58baafe8-7b37-11eb-071c-0fb7dd06e885
md"""
**Quiz Question:** Is the validation error of the new decision tree (using early stopping conditions 2 and 3) lower than, higher than, or the same as that of the old decision tree from the previous assignment?
  - should be lower! (pretty close)...
"""

# ╔═╡ 7bb705f0-7b37-11eb-0a51-851d7325ad47
md"""
### Exploring the effect of max_depth

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**too small**, **just right**, and **too large**).

Train three models with these parameters:

1. **model_1**: max_depth = 2 (too small)
2. **model_2**: max_depth = 6 (just right)
3. **model_3**: max_depth = 14 (may be too large)

For each of these three, we set `min_node_size = 0` and `min_error_reduction = -1`.

**Note:** Each tree can take up to a few minutes to train. In particular, `model_3` will probably take the longest to train.

"""

# ╔═╡ 7ba0b250-7b37-11eb-08e3-635d3737c018
begin
	model_1 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=2, min_node_size=0, min_error_reduction=-1)
	model_2 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6, min_node_size=0, min_error_reduction=-1)
	model_3 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=14, min_node_size=0, min_error_reduction=-1)
end

# ╔═╡ 7b815040-7b37-11eb-0a69-8fbbd0859786
md"""
#### Evaluating the models

Let us evaluate the models on the train and validation data. Let us start by evaluating the classification error on the training data:
"""

# ╔═╡ 7b68e74e-7b37-11eb-05df-8b16a1439d68
function print_eval(models, df, target;
		df_label=:training, offset=0, digits=4)
	for (ix, m) ∈ enumerate(models)
		cr = eval_classification_error(m, df, target)
		@printf("%10s data, classification error (model %2d): %1.4f\n", 
			df_label, ix + offset, round(cr; digits=digits))
	end
end

# ╔═╡ 7b54fec0-7b37-11eb-37bc-7df8411fb645
with_terminal() do
	print_eval([model_1, model_2, model_3], train_data, Target)
end

# ╔═╡ 7b25f5d8-7b37-11eb-30dd-677a45cd9920
md"""
Now evaluate the classification error on the validation data.
"""

# ╔═╡ 69b44c90-7b38-11eb-14c6-3dd9dd0d2303
with_terminal() do
	print_eval([model_1, model_2, model_3], validation_data, Target)
end

# ╔═╡ 6993ba5c-7b38-11eb-31b8-69bdfe81ed69
md"""
**Quiz Question:** Which tree has the smallest error on the validation data?
  - model2 (in this case, with this sample) 

**Quiz Question:** Does the tree with the smallest error in the training data also have the smallest error in the validation data?
  - no (in this case, with this sample)

**Quiz Question:** Is it always true that the tree with the lowest classification error on the **training** set will result in the lowest classification error in the **validation** set?
  - no


### Measuring the complexity of the tree

Recall in the lecture that we talked about deeper trees being more complex. We will measure the complexity of the tree as

```
  complexity(T) = number of leaves in the tree T
```
"""

# ╔═╡ 6974e686-7b38-11eb-2ad1-017f6ac940dd
md"""
Compute the number of nodes in model\_1, model\_2, and model\_3.
"""

# ╔═╡ cccf9118-7b38-11eb-2f79-f9575218f0a5
with_terminal() do
	for (ix, m) ∈ enumerate([model_1, model_2, model_3])
		cn = count_leaves(m)
  		@printf("for model %2d - num of node is: %4d\n", ix, cn)
	end
end

# ╔═╡ ccb33ee6-7b38-11eb-072e-f31cc84a982a
md"""
**Quiz Question:** Which tree has the largest complexity?
  - model 3

**Quiz Question:** Is it always true that the most complex tree will result in the lowest classification error in the **validation_set**?
  - no (most likely it will overfit the training set, and wil generalize poorly).
"""

# ╔═╡ cc976a84-7b38-11eb-12ec-a5cf5ea3c4c1
md"""
### Exploring the effect of min_error

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**negative**, **just right**, and **too positive**).

Train three models with these parameters:
  1. **model_4**: `min_error_reduction = -1` (ignoring this early stopping condition)
  1. **model_5**: `min_error_reduction = 0` (just right)
  1. **model_6**: `min_error_reduction = 5` (too positive)

For each of these three, we set `max_depth = 6`, and `min_node_size = 0`.
"""

# ╔═╡ cc7c625e-7b38-11eb-1bbb-e39f6ba35768
begin
	model_4 = decision_tree_create(train_data, features, :safe_loans; 
		max_depth=6, min_node_size=0, min_error_reduction=-1)
	
	model_5 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6,  min_node_size=0, min_error_reduction=0)
	
	model_6 = decision_tree_create(train_data, features, :safe_loans; 
		max_depth=6, min_node_size=0, min_error_reduction=5)
end

# ╔═╡ 695bb3dc-7b38-11eb-3b98-bd0e15b4de11
md"""
Calculate the accuracy of each model (model\_4, model\_5, or model\_6) on the validation set. 
"""

# ╔═╡ fca4be14-7b39-11eb-2890-7577dda5d4de
with_terminal() do
	print_eval([model_4, model_5, model_6], validation_data, Target;
	df_label=:validation, offset=4)
end

# ╔═╡ fca323ea-7b39-11eb-34ff-9b3613ca036c
md"""
Using the count_leaves function, compute the number of leaves in each of each models in (model\_4, model\_5, and model\_6). 
"""

# ╔═╡ fca2a480-7b39-11eb-1d64-d7e7ff02159c
with_terminal() do
	for (ix, m) in enumerate([model_4, model_5, model_6])
  		cn = count_leaves(m)
  		@printf("for model %2d - num of node is: %4d\n", ix, cn)
	end
end

# ╔═╡ 41f440dc-7b3a-11eb-08e3-635d3737c018
md"""
**Quiz Question:** Using the complexity definition above, which model (**model_4**, **model_5**, or **model_6**) has the largest complexity?
  - model_4 and model_5

  Did this match your expectation?
  - pretty much
  
**Quiz Question:** **model_4** and **model_5** have similar classification error on the validation set but **model_5** has lower complexity. Should you pick **model_5** over **model_4**?
  - yes
"""

# ╔═╡ 41d5b2ca-7b3a-11eb-0a69-8fbbd0859786
md"""
### Exploring the effect of min_node_size

We will compare three models trained with different values of the stopping criterion. Again, intentionally picked models at the extreme ends (**too small**, **just right**, and **just right**).

Train three models with these parameters:
  1. **model_7**: min_node_size = 0 (too small)
  2. **model_8**: min_node_size = 2000 (just right)
  3. **model_9**: min_node_size = 50000 (too large)

For each of these three, we set `max_depth = 6`, and `min_error_reduction = -1`.
"""

# ╔═╡ 41bd2950-7b3a-11eb-05df-8b16a1439d68
begin
	model_7 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6, min_node_size=0, min_error_reduction=-1);
	
	model_8 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6, min_node_size=2000, min_error_reduction=-1);
	
	model_9 = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6, min_node_size=50000, min_error_reduction=-1)
end

# ╔═╡ 419fec1c-7b3a-11eb-37bc-7df8411fb645
md"""
Now, let us evaluate the models (model\_7, model\_8, or model\_9) on the validation_set.
"""

# ╔═╡ b571a75c-7b3a-11eb-3b98-bd0e15b4de11
with_terminal() do
	print_eval([model_7, model_8, model_9], validation_data, Target; 	df_label=:validation, offset=6)
end

# ╔═╡ b5553504-7b3a-11eb-12fa-4d42c67876b2
md"""
Using the count_leaves function, compute the number of leaves in each of each models (model\_7, model\_8, and model\_9). 
"""

# ╔═╡ b539906a-7b3a-11eb-20d9-87257af20d51
with_terminal() do
	for (ix, m) ∈ enumerate([model_7, model_8, model_9])
  		cn = count_leaves(m)
  		@printf("for model %2d - num of node is: %4d\n", ix, cn)
	end
end

# ╔═╡ b51dd136-7b3a-11eb-0a51-851d7325ad47
md"""
**Quiz Question:** Using the results obtained in this section, which model (**model_7**, **model_8**, or **model_9**) would you choose to use?
  - model_8 (although not clear cut!)
"""

# ╔═╡ Cell order:
# ╟─5435ff66-7a2b-11eb-1474-b96dcad21315
# ╠═77311686-7a2b-11eb-2bc7-95b2a2211969
# ╠═0d124834-7ef3-11eb-1d64-d7e7ff02159c
# ╟─771130dc-7a2b-11eb-170f-934c11fceede
# ╠═76f62e40-7a2b-11eb-095e-f36e207f06a2
# ╟─76de16d4-7a2b-11eb-09c6-e36b6c893c1f
# ╠═76c17c4a-7a2b-11eb-2c47-97106bd58f99
# ╟─76a83684-7a2b-11eb-1961-cb133db8c164
# ╠═768d6b08-7a2b-11eb-388a-8763c8da0a51
# ╟─7673d8e4-7a2b-11eb-02b4-2f837181b4e9
# ╠═1f459740-7ef3-11eb-34ff-9b3613ca036c
# ╠═32a93648-7ef3-11eb-071c-0fb7dd06e885
# ╠═31f2811e-7ef3-11eb-2890-7577dda5d4de
# ╟─760c171a-7a2b-11eb-1341-19ba5a1824a1
# ╠═75d98df2-7a2b-11eb-296e-ef4d312ffcfa
# ╠═75c1a64e-7a2b-11eb-21f6-55dd3a8ce164
# ╠═75a52c44-7a2b-11eb-2f02-2312f3a3dac9
# ╠═75736a9c-7a2b-11eb-2278-ab37447cf22d
# ╠═116eb5f8-7a33-11eb-2c4c-5f937540f987
# ╟─a5c56d68-7a32-11eb-2ef6-e58ed65dc561
# ╠═a5a1cdf6-7a32-11eb-18dd-0dfee8eb885c
# ╟─a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
# ╠═6fad8afe-7a33-11eb-3208-6b289c3ccd23
# ╟─6f88e4f6-7a33-11eb-10cf-e3e8608e616b
# ╟─2e859492-7b2b-11eb-1d64-d7e7ff02159c
# ╠═49f07e5c-7b2b-11eb-34ff-9b3613ca036c
# ╟─84eab9ca-7b2b-11eb-2890-7577dda5d4de
# ╟─94fccdee-7b2b-11eb-071c-0fb7dd06e885
# ╠═a58dfb04-7b2b-11eb-30dd-677a45cd9920
# ╟─c4c23a3c-7b2b-11eb-37bc-7df8411fb645
# ╟─ce3be5b8-7b2b-11eb-1bbb-e39f6ba35768
# ╠═6f3011da-7a33-11eb-2cd0-b7bb04b7b545
# ╟─ce1d2376-7b2b-11eb-14c6-3dd9dd0d2303
# ╠═ce025442-7b2b-11eb-31b8-69bdfe81ed69
# ╟─cde76c36-7b2b-11eb-2ad1-017f6ac940dd
# ╟─cdcdcc7c-7b2b-11eb-3b98-bd0e15b4de11
# ╠═b6fc15ae-7a37-11eb-071c-0fb7dd06e885
# ╟─cdb69d9c-7b2b-11eb-12fa-4d42c67876b2
# ╠═cd9e2b4a-7b2b-11eb-20d9-87257af20d51
# ╟─753b3df2-7a2b-11eb-28c0-b5bdffef414b
# ╠═be0730f6-7a3d-11eb-30dd-677a45cd9920
# ╟─a92a0af0-7b34-11eb-2890-7577dda5d4de
# ╠═c88e3ac2-7b34-11eb-0a69-8fbbd0859786
# ╟─c872733c-7b34-11eb-05df-8b16a1439d68
# ╟─c85ab936-7b34-11eb-37bc-7df8411fb645
# ╠═c8405834-7b34-11eb-30dd-677a45cd9920
# ╠═594b159e-7b35-11eb-071c-0fb7dd06e885
# ╠═59304232-7b35-11eb-2890-7577dda5d4de
# ╟─59141350-7b35-11eb-34ff-9b3613ca036c
# ╠═aca4b40c-7b35-11eb-20d9-87257af20d51
# ╟─ac83767a-7b35-11eb-0a51-851d7325ad47
# ╟─ac686542-7b35-11eb-08e3-635d3737c018
# ╠═ac4e3b42-7b35-11eb-0a69-8fbbd0859786
# ╟─c825a994-7b34-11eb-071c-0fb7dd06e885
# ╠═3bf28f8e-7b37-11eb-2890-7577dda5d4de
# ╟─58baafe8-7b37-11eb-071c-0fb7dd06e885
# ╟─7bb705f0-7b37-11eb-0a51-851d7325ad47
# ╠═7ba0b250-7b37-11eb-08e3-635d3737c018
# ╟─7b815040-7b37-11eb-0a69-8fbbd0859786
# ╠═7b68e74e-7b37-11eb-05df-8b16a1439d68
# ╠═7b54fec0-7b37-11eb-37bc-7df8411fb645
# ╟─7b25f5d8-7b37-11eb-30dd-677a45cd9920
# ╠═69b44c90-7b38-11eb-14c6-3dd9dd0d2303
# ╟─6993ba5c-7b38-11eb-31b8-69bdfe81ed69
# ╟─6974e686-7b38-11eb-2ad1-017f6ac940dd
# ╠═cccf9118-7b38-11eb-2f79-f9575218f0a5
# ╟─ccb33ee6-7b38-11eb-072e-f31cc84a982a
# ╟─cc976a84-7b38-11eb-12ec-a5cf5ea3c4c1
# ╠═cc7c625e-7b38-11eb-1bbb-e39f6ba35768
# ╟─695bb3dc-7b38-11eb-3b98-bd0e15b4de11
# ╠═fca4be14-7b39-11eb-2890-7577dda5d4de
# ╟─fca323ea-7b39-11eb-34ff-9b3613ca036c
# ╠═fca2a480-7b39-11eb-1d64-d7e7ff02159c
# ╟─41f440dc-7b3a-11eb-08e3-635d3737c018
# ╟─41d5b2ca-7b3a-11eb-0a69-8fbbd0859786
# ╠═41bd2950-7b3a-11eb-05df-8b16a1439d68
# ╟─419fec1c-7b3a-11eb-37bc-7df8411fb645
# ╠═b571a75c-7b3a-11eb-3b98-bd0e15b4de11
# ╟─b5553504-7b3a-11eb-12fa-4d42c67876b2
# ╠═b539906a-7b3a-11eb-20d9-87257af20d51
# ╟─b51dd136-7b3a-11eb-0a51-851d7325ad47
