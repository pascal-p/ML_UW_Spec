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

# ╔═╡ 3578d030-7eef-11eb-1d64-d7e7ff02159c
begin
	const DF = AbstractDataFrame # Union{DataFrame, SubDataFrame}
	
	include("./utils.jl");	
	include("./dt_utils.jl");
end

# ╔═╡ 5435ff66-7a2b-11eb-1474-b96dcad21315
md"""
## C03w03: Implementing binary Decision Trees

The goal of this notebook is to implement your own binary decision tree classifier. We will:
    
  - Use DataFrames to do some feature engineering.
  - Transform categorical variables into binary variables.
  - Write a function to compute the number of misclassified examples in an intermediate node.
  - Write a function to find the best feature to split on.
  - Build a binary decision tree from scratch.
  - Make predictions using the decision tree.
  - Evaluate the accuracy of the decision tree.
  - Visualize the decision at the root node.

**Important Note**: In this assignment, we will focus on building decision trees where the data contain **only binary (0 or 1) features**. This allows us to avoid dealing with:
  -  Multiple intermediate nodes in a split
  - The thresholding issues of real-valued features.
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

# ╔═╡ dde7e8e4-7a2d-11eb-2527-3df1ab7d0796
describe(loans, :eltype, :nmissing, :first => first)

# ╔═╡ 7673d8e4-7a2b-11eb-02b4-2f837181b4e9
md"""
#### Subsample dataset to make sure classes are balanced

Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points.
"""

# ╔═╡ 542db682-7a2e-11eb-35ad-25c82d6919e0
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

# ╔═╡ c5038900-7eec-11eb-34ff-9b3613ca036c
loans_data = resample!(safe_loans₀, risky_loans₀);

# ╔═╡ 3d3614d8-7a2e-11eb-2a8a-b53822e7928e
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
	hot_encode!(loans_data);
	loans_data[1:5, :];
end

# ╔═╡ 75a52c44-7a2b-11eb-2f02-2312f3a3dac9
begin
	features = names(loans_data) |> a_ -> Symbol.(a_);
	deleteat!(features, 
		findall(x -> x == Target, features)) 
end

# ╔═╡ 44158e34-7a40-11eb-1d64-d7e7ff02159c
first(loans_data, 3)

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
	train_data, test_data = train_test_split(loans_data; split=0.8, seed=42);
	size(train_data), size(test_data)
    ## vs ((37224, 26), (9284, 26)) ...
end

# ╔═╡ 6f88e4f6-7a33-11eb-10cf-e3e8608e616b
md"""
### Decision tree implementation

In this section, we will implement binary decision trees from scratch. There are several steps involved in building a decision tree. For that reason, we have split the entire assignment into several sections.

#### Function to count number of mistakes while predicting majority class

Recall from the lecture that prediction at an intermediate node works by predicting the **majority class** for all data points that belong to this node.

Now, we will write a function that calculates the number of **missclassified examples** when predicting the **majority class**. This will be used to help determine which feature is the best to split on at a given node of the tree.

**Note**: Keep in mind that in order to compute the number of mistakes for a majority classifier, we only need the label (y values) of the data points in the node. 

**Steps to follow**:
  1. Calculate the number of safe loans and risky loans.
  1. Since we are assuming majority class prediction, all the data points that are **not** in the majority class are considered **mistakes**.
  1. Return the number of **mistakes**.


Now, let us write the function `inter_node_num_mistakes` which computes the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node.
"""

# ╔═╡ 6f6802b8-7a33-11eb-0c47-5d8b41dcfb39
function inter_node_num_mistakes(labels_in_node::AbstractVector{T}) where {T <: Real}
    length(labels_in_node) == 0 && return 0  ## Corner case

	n = length(labels_in_node) 
	n_pos = sum(labels_in_node .== 1)  ## Count the number of 1's (safe loans)
    n_neg = n - n_pos
	
	## Return the number of mistakes that the majority classifier makes.
	return n_pos ≥ n_neg ? n_neg : n_pos
end

# ╔═╡ 6f4a5e40-7a33-11eb-038f-438ba7d14f44
md"""
Because there are several steps in this assignment, we have introduced some stopping points where you can check your code and make sure it is correct before proceeding.
To test your inter_node_num_mistakes function, run the following code until you get a Test passed!, then you should proceed. Otherwise, you should spend some time figuring out where things went wrong.
"""

# ╔═╡ 6f3011da-7a33-11eb-2cd0-b7bb04b7b545
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

# ╔═╡ 6f177b72-7a33-11eb-0972-f35a77f51499
md"""
#### Function to pick best feature to split on

The function `best_splitting_feature` takes 3 arguments: 
1. The data (DataFrame which includes all of the feature columns and label column)
2. The features to consider for splits (a list of Symbol of column names to consider for splits)
3. The name of the target/label column (Symbol)

The function will loop through the list of possible features, and consider splitting on each of them. It will calculate the classification error of each split and return the feature that had the smallest classification error when split on.

Recall that the **classification error** is defined as follows:

$$\mbox{classification error} = \frac{\mbox{\# mistakes}}{\mbox{\# total examples}}$$

Follow these steps: 
  - **Step 1:** Loop over each feature in the feature list
  -  **Step 2:** Within the loop, split the data into two groups: one group where all of the data has feature value 0 or False (we will call this the **left** split), and one group where all of the data has feature value 1 or True (we will call this the **right** split). Make sure the **left** split corresponds with 0 and the **right** split corresponds with 1 to ensure your implementation fits with our implementation of the tree building process.
  - **Step 3:** Calculate the number of misclassified examples in both groups of data and use the above formula to compute the **classification error**.
  - **Step 4:** If the computed error is smaller than the best error found so far, store this **feature and its error**.

This may seem like a lot, but we have provided pseudocode in the comments in order to help you implement the function correctly.

**Note:** Remember that since we are only dealing with binary features, we do not have to consider thresholds for real-valued features. This makes the implementation of this function much easier.
"""

# ╔═╡ 6efa48f4-7a33-11eb-1dac-3bf64871bf99
function best_splitting_feature(df::DF, features::Vector{Symbol}, 
		target::Symbol) where {DF <: Union{DataFrame, SubDataFrame}}
	best_feature, best_error = nothing, 2
	## Note: Since error is always ≤ 1, init best_error with something > 1
	
	num_data_points = size(df)[1]
	for f ∈ features
		l_split = df[df[!, f] .== 0, :]
		r_split = df[df[!, f] .== 1, :]
		
		l_mistakes = inter_node_num_mistakes(l_split[!, target])
		r_mistakes = inter_node_num_mistakes(r_split[!, target])
		
		error = (l_mistakes + r_mistakes) / num_data_points
		if error < best_error
			best_error = error
          	best_feature = f
		end
	end
	
	best_feature
end

# ╔═╡ 57fb569a-7a36-11eb-1d64-d7e7ff02159c
@test best_splitting_feature(train_data, features, Target) == Symbol("term_ 60 months")

# ╔═╡ 6ee0a5a2-7a33-11eb-31db-d164581fd26a
md"""
#### Building the tree

With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:

```julia
    ( 
       :is_leaf            => true/false.
       :prediction         => Prediction at the leaf node.
       :left               => (dictionary corresponding to the left tree).
       :right              => (dictionary corresponding to the right tree).
       :splitting_feature  => The feature that this node splits on.
    )
```

First, we will write a function that creates a leaf node given a set of target values. 
"""

# ╔═╡ a553b5ee-7a32-11eb-329a-419005d73e2c
function create_leaf(target_values; 
		splitting_feature=nothing, 
		left=nothing, right=nothing, is_leaf=true)

    ## Create a leaf node
    leaf = Dict{Symbol, Any}(
      :splitting_feature => splitting_feature,
	  :prediction => nothing,
      :left => left,
      :right => right,
      :is_leaf => is_leaf,
    )  
    
	if is_leaf
    	## Count the number of data points that are +1 and -1 in this node.
    	num_1s = length(target_values[target_values .== 1, :])
    	num_minus_1s = length(target_values[target_values .== -1, :])
    
    	## For the leaf node, set the prediction to be the majority class.
    	## Store the predicted class (1 or -1) in leaf['prediction']
    	leaf[:prediction] = num_1s > num_minus_1s ? 1 : -1
	end
    
	leaf
end

# ╔═╡ b715aca6-7a37-11eb-30dd-677a45cd9920
md"""
We have provided a function that learns the decision tree recursively and implements 3 stopping conditions:
  1. **Stopping condition 1:** All data points in a node are from the same class.
  2. **Stopping condition 2:** No more features to split on.
  3. **Additional stopping condition:** In addition to the above two stopping conditions covered in lecture, in this assignment we will also consider a stopping condition based on the **max_depth** of the tree. By not letting the tree grow too deep, we will save computational effort in the learning process. 

Now, we will write down the skeleton of the learning algorithm. 
"""

# ╔═╡ b6fc15ae-7a37-11eb-071c-0fb7dd06e885
function decision_tree_create(df::DF, features, target;
		curr_depth=0, max_depth=10, 
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
        verbose && @printf("Reached maximum depth. Stopping for now.")
        return create_leaf(target_vals)
	end

    ## Now Find the best splitting feature 
    splitting_feature = best_splitting_feature(df, features, target)
    
    # Split on the best feature that we found. 
    l_split = df[df[!, splitting_feature] .== 0, :]
    r_split = df[df[!, splitting_feature] .== 1, :]
	
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
						curr_depth=curr_depth + 1, 
						max_depth, verbose)
    rtree = decision_tree_create(r_split, rem_features, target;
						curr_depth=curr_depth + 1, 
						max_depth, verbose)

    return create_leaf(target_vals; 
						splitting_feature=string(splitting_feature), 
						left=ltree, 
						right=rtree, 
						is_leaf=false)		
end

# ╔═╡ ae0b1116-7a3a-11eb-37bc-7df8411fb645
# function count_nodes(tree)
#     isnothing(tree) || tree[:is_leaf] ? 
# 	  1 : 1 + count_nodes(tree[:left]) + count_nodes(tree[:right])
# end

# ╔═╡ b6e1567c-7a37-11eb-2890-7577dda5d4de
md"""

Run the following test code to check the implementation. 
"""

# ╔═╡ b6c80f78-7a37-11eb-34ff-9b3613ca036c
begin
	small_data_decision_tree = decision_tree_create(train_data, features, :safe_loans;
		max_depth=3, verbose=true)

	@test count_nodes(small_data_decision_tree) == 13
end

# ╔═╡ 753b3df2-7a2b-11eb-28c0-b5bdffef414b
md"""
### Build the tree!

Now that all the tests are passing, we will train a tree model on the train_data. Limit the depth to 6 `max_depth = 6` to make sure the algorithm doesn't run for too long. 

Call this tree `dt6_model`.

"""

# ╔═╡ be0730f6-7a3d-11eb-30dd-677a45cd9920
dt6_model = decision_tree_create(train_data, features, :safe_loans;
		max_depth=6)

# ╔═╡ bdeea6bc-7a3d-11eb-071c-0fb7dd06e885
md"""
### Making predictions with a decision tree

As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. Below, we call this function `classify` (cf. `dt_utils.jl`), which takes in a learned tree and a test point x to classify. We include an option annotate that describes the prediction path when set to True.

"""

# ╔═╡ bdb7d658-7a3d-11eb-34ff-9b3613ca036c
md"""
Now, let's consider the first example of the test set and see what `dt6_model`  predicts for this data point.
"""

# ╔═╡ e82bb3ce-7a3e-11eb-1d64-d7e7ff02159c
with_terminal() do
	@printf("Predicted class: %s\n", classify(dt6_model, test_data[1, :]; 
			annotate=true))
end

# ╔═╡ 96aa449a-7a4b-11eb-37bc-7df8411fb645
md"""
**Quiz Question**: What was the feature that dt6_model first split on while making the prediction for `test_data[1, :]`?
  - "term 60 months"

**Quiz Question**: What was the first feature that lead to a right split of `test_data[1, :]`?
  - "gradeF" (where split\_feature\_value was 1)

**Quiz Question**: What was the last feature split on before reaching a leaf node for `test_data[1, :]`?
  - "gradeF"

"""

# ╔═╡ 94f1a986-7a4b-11eb-30dd-677a45cd9920
md"""
#### Evaluating your decision tree

Now, we will write a function to evaluate a decision tree by computing the classification error of the tree on the given dataset.

Again, recall that the **classification error** is defined as follows:

$$\mbox{classification error} = \frac{\mbox{\# mistakes}}{\mbox{\# total examples}}$$

Now, write a function called `evaluate_classification_error` that takes in as input:
1. `tree` (as described above)
2. `data` (an SFrame)
3. `target` (a string - the name of the target/label column)

This function should calculate a prediction (class label) for each row in `data` using the decision `tree` and return the classification error computed using the above formula.
"""

# ╔═╡ 7b7e1d1c-7a4c-11eb-0a51-851d7325ad47
function eval_classification_error(tree, df::DF, target::Symbol) where {DF <: Union{DataFrame, SubDataFrame}}
	## 1. get the predictions
    ŷ = map(x -> classify(tree, x), eachrow(df))
	
    ## 2. calculate the classification error
    num_mistakes = sum(ŷ .≠ df[!, target])
    return num_mistakes / size(df)[1]
end

# ╔═╡ 7b66e926-7a4c-11eb-08e3-635d3737c018
begin
	error = eval_classification_error(dt6_model, test_data, Target)
	(class_error=round(error, digits=2), 
		class_accuracy=round(1. - error, digits=2))
end

# ╔═╡ 7b4d4fb6-7a4c-11eb-0a69-8fbbd0859786
md"""
**Quiz Question**: Rounded to 2nd decimal point, what is the classification error of `dt6_model` on the test\_data?
  - cf. above cell.
"""

# ╔═╡ 7b31c35e-7a4c-11eb-05df-8b16a1439d68
md"""
#### Printing out a decision stump

As discussed in the lecture, we can print out a single decision stump . 
"""

# ╔═╡ 881bdbb2-7a57-11eb-20d9-87257af20d51
function  print_stump(tree, name=:root)
    split_name = tree[:splitting_feature] # ex. 'term. 36 months'
	
    if isnothing(split_name)
        print("(leaf, label: $(tree[:prediction]))")
        return nothing
	end
	# println("==> ", split_name)
    # split_feature, split_value = split(split_name, "_")
    println("                       $(name)")
    println("         |---------------|----------------|")
    println("         |                                |")
    println("         |                                |")
    println("         |                                |")
    println("     [$(split_name) == 0]              [$(split_name) == 1]")
    println("         |                                |")
    println("         |                                |")
    println("         |                                |")
    @printf("     (%s)                     (%s)\n",
        tree[:left][:is_leaf] ? 
			string("leaf, label: ", string(tree[:left][:prediction])) : "subtree",
		tree[:right][:is_leaf] ? 
        	string("leaf, label: ", string(tree[:right][:prediction])) : "subtree")
end

# ╔═╡ caea5eb8-7a58-11eb-12fa-4d42c67876b2
with_terminal() do
	print_stump(dt6_model)
end

# ╔═╡ 88b9603c-7a5a-11eb-2ad1-017f6ac940dd
with_terminal() do
	dt6_model[:splitting_feature]
end

# ╔═╡ aca01ac2-7a5a-11eb-072e-f31cc84a982a
with_terminal() do
	print_stump(dt6_model[:left], dt6_model[:splitting_feature])
end

# ╔═╡ ac83442e-7a5a-11eb-12ec-a5cf5ea3c4c1
with_terminal() do
	print_stump(dt6_model[:left][:left][:left], 
		dt6_model[:left][:left][:splitting_feature])
end

# ╔═╡ ac6589fc-7a5a-11eb-1bbb-e39f6ba35768
with_terminal() do
	print_stump(dt6_model[:right][:right], dt6_model[:right][:splitting_feature])
end

# ╔═╡ Cell order:
# ╟─5435ff66-7a2b-11eb-1474-b96dcad21315
# ╠═77311686-7a2b-11eb-2bc7-95b2a2211969
# ╠═3578d030-7eef-11eb-1d64-d7e7ff02159c
# ╟─771130dc-7a2b-11eb-170f-934c11fceede
# ╠═76f62e40-7a2b-11eb-095e-f36e207f06a2
# ╟─76de16d4-7a2b-11eb-09c6-e36b6c893c1f
# ╠═76c17c4a-7a2b-11eb-2c47-97106bd58f99
# ╟─76a83684-7a2b-11eb-1961-cb133db8c164
# ╠═768d6b08-7a2b-11eb-388a-8763c8da0a51
# ╠═dde7e8e4-7a2d-11eb-2527-3df1ab7d0796
# ╟─7673d8e4-7a2b-11eb-02b4-2f837181b4e9
# ╠═542db682-7a2e-11eb-35ad-25c82d6919e0
# ╠═c5038900-7eec-11eb-34ff-9b3613ca036c
# ╠═3d3614d8-7a2e-11eb-2a8a-b53822e7928e
# ╟─760c171a-7a2b-11eb-1341-19ba5a1824a1
# ╠═75d98df2-7a2b-11eb-296e-ef4d312ffcfa
# ╠═75c1a64e-7a2b-11eb-21f6-55dd3a8ce164
# ╠═75a52c44-7a2b-11eb-2f02-2312f3a3dac9
# ╠═44158e34-7a40-11eb-1d64-d7e7ff02159c
# ╠═75736a9c-7a2b-11eb-2278-ab37447cf22d
# ╠═116eb5f8-7a33-11eb-2c4c-5f937540f987
# ╟─a5c56d68-7a32-11eb-2ef6-e58ed65dc561
# ╠═a5a1cdf6-7a32-11eb-18dd-0dfee8eb885c
# ╟─a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
# ╠═6fad8afe-7a33-11eb-3208-6b289c3ccd23
# ╟─6f88e4f6-7a33-11eb-10cf-e3e8608e616b
# ╠═6f6802b8-7a33-11eb-0c47-5d8b41dcfb39
# ╟─6f4a5e40-7a33-11eb-038f-438ba7d14f44
# ╠═6f3011da-7a33-11eb-2cd0-b7bb04b7b545
# ╟─6f177b72-7a33-11eb-0972-f35a77f51499
# ╠═6efa48f4-7a33-11eb-1dac-3bf64871bf99
# ╠═57fb569a-7a36-11eb-1d64-d7e7ff02159c
# ╟─6ee0a5a2-7a33-11eb-31db-d164581fd26a
# ╠═a553b5ee-7a32-11eb-329a-419005d73e2c
# ╟─b715aca6-7a37-11eb-30dd-677a45cd9920
# ╠═b6fc15ae-7a37-11eb-071c-0fb7dd06e885
# ╠═ae0b1116-7a3a-11eb-37bc-7df8411fb645
# ╟─b6e1567c-7a37-11eb-2890-7577dda5d4de
# ╠═b6c80f78-7a37-11eb-34ff-9b3613ca036c
# ╟─753b3df2-7a2b-11eb-28c0-b5bdffef414b
# ╠═be0730f6-7a3d-11eb-30dd-677a45cd9920
# ╟─bdeea6bc-7a3d-11eb-071c-0fb7dd06e885
# ╟─bdb7d658-7a3d-11eb-34ff-9b3613ca036c
# ╠═e82bb3ce-7a3e-11eb-1d64-d7e7ff02159c
# ╟─96aa449a-7a4b-11eb-37bc-7df8411fb645
# ╟─94f1a986-7a4b-11eb-30dd-677a45cd9920
# ╠═7b7e1d1c-7a4c-11eb-0a51-851d7325ad47
# ╠═7b66e926-7a4c-11eb-08e3-635d3737c018
# ╟─7b4d4fb6-7a4c-11eb-0a69-8fbbd0859786
# ╟─7b31c35e-7a4c-11eb-05df-8b16a1439d68
# ╠═881bdbb2-7a57-11eb-20d9-87257af20d51
# ╠═caea5eb8-7a58-11eb-12fa-4d42c67876b2
# ╠═88b9603c-7a5a-11eb-2ad1-017f6ac940dd
# ╠═aca01ac2-7a5a-11eb-072e-f31cc84a982a
# ╠═ac83442e-7a5a-11eb-12ec-a5cf5ea3c4c1
# ╠═ac6589fc-7a5a-11eb-1bbb-e39f6ba35768
