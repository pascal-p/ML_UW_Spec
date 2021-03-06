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
  using Plots
end

# ╔═╡ e9f8e942-7b29-11eb-1d64-d7e7ff02159c
begin
  include("./utils.jl");
  include("./dt_utils.jl");
end

# ╔═╡ 5435ff66-7a2b-11eb-1474-b96dcad21315
md"""
## C03w05: Boosting a decision stump

The goal of this notebook is to implement our own boosting module.

  - Use SFrames to do some feature engineering.
  - Modify the decision trees to incorporate weights.
  - Implement Adaboost ensembling.
  - Use your implementation of Adaboost to train a boosted decision stump ensemble.
  - Evaluate the effect of boosting (adding more decision stumps) on performance of the model.
  - Explore the robustness of Adaboost to overfitting.
"""

# ╔═╡ 81b7a9e0-7cc8-11eb-1d64-d7e7ff02159c
const DF= Union{DataFrame, SubDataFrame}

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
#### Extracting the target and the feature columns

We will now repeat some of the feature processing steps that we saw in the previous assignment:

Next, we select four categorical features:
  1. grade of the loan
  2. the length of the loan term
  3. the home ownership status: own, mortgage, rent
  4. number of years of employment.
"""

# ╔═╡ 768d6b08-7a2b-11eb-388a-8763c8da0a51
begin
    const Features = [
    	:grade,           # grade of the loan
        :emp_length,      # number of years of employment
        :home_ownership,  # home_ownership status: own, mortgage or rent
        :term,            # the term of the loan
    ]

  Target = :safe_loans # prediction target (y) (+1 means safe, -1 is risky)

  # Extract the feature columns and target column
  select!(loans, [Features..., Target]);
  length(names(loans)), names(loans)
end

# ╔═╡ 7673d8e4-7a2b-11eb-02b4-2f837181b4e9
md"""
#### Subsample dataset to make sure classes are balanced

Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points.
"""

# ╔═╡ 3773355e-7cc8-11eb-2890-7577dda5d4de
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

# ╔═╡ 6327985a-7cc8-11eb-071c-0fb7dd06e885
loans_data = resample!(safe_loans₀, risky_loans₀);

# ╔═╡ 7641d404-7a2b-11eb-042d-bbc2c500ac59
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
  train_data, test_data = train_test_split(loans_data; split=0.8, seed=70);
  size(train_data), size(test_data)
    ## vs ((37224, 26), (9284, 26)) ...
end

# ╔═╡ 6f88e4f6-7a33-11eb-10cf-e3e8608e616b
md"""
### Weighted decision trees

Let's modify our decision tree code from Module 5 to support weighting of individual data points.

#### Weighted error definition

Consider a model with $N$ data points with:
  -  Predictions $\hat{y}_1 ... \hat{y}_n$
  -  Target $y_1 ... y_n$
  -  Data point weights $\alpha_1 ... \alpha_n$.

Then the **weighted error** is defined by:

$$\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}]}{\sum_{i=1}^{n} \alpha_i}$$

where $1[y_i \neq \hat{y_i}]$ is an indicator function that is set to $1$ if $y_i \neq \hat{y_i}$.


#### Write a function to compute weight of mistakes

Write a function that calculates the weight of mistakes for making the "weighted-majority" predictions for a dataset. The function accepts two inputs:
  -  `labels_in_node`: Targets $y_1 ... y_n$
  -  `data_weights`: Data point weights $\alpha_1 ... \alpha_n$

We are interested in computing the (total) weight of mistakes, i.e.

$$\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}].$$

This quantity is analogous to the number of mistakes, except that each mistake now carries different weight. It is related to the weighted error in the following way:

$$\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\sum_{i=1}^{n} \alpha_i}$$

The function `inter_node_weighted_mistakes` should first compute two weights:
   - weight of mistakes $\mathrm{WM}_{-1}$, when all predictions are
$\hat{y}_i = -1,\ i.e\ \mathrm{WM}(\mathbf{\alpha}, \mathbf{-1})$

   -  weight of mistakes $\mathrm{WM}_{+1}$ when all predictions are
$\hat{y}_i = +1,\ i.e\ \mbox{WM}(\mathbf{\alpha}, \mathbf{+1})$

 where $\mathbf{-1}$ and $\mathbf{+1}$ are vectors where all values are -1 and +1 respectively.

After computing $\mathrm{WM}_{-1}$ and $\mathrm{WM}_{+1}$, the function `inter_node_weighted_mistakes` should return the lower of the two weights of mistakes, along with the class associated with that weight.
"""

# ╔═╡ 49f07e5c-7b2b-11eb-34ff-9b3613ca036c
function inter_node_weighted_mistakes(labels_in_node::AbstractVector{T},
                                      data_weights) where {T <: Real}

  ## Sum the weights of all +1 entries ≡ Weight of mistakes for predicting all -1's
  w_mistakes_all_neg = sum(data_weights[labels_in_node .== 1])

  ## Sum the weights of all -1 entries ≡ Weight of mistakes for predicting all +1's
  w_mistakes_all_pos = sum(data_weights[labels_in_node .== -1])

  ## Return the tuple (weight, class_label) representing the lower of the two weights.
  ## class_label should be an integer of value +1 or -1. tie => (tot_weight_po, +1)
  return w_mistakes_all_pos ≤ w_mistakes_all_neg ? (w_mistakes_all_pos, 1) :
  (w_mistakes_all_neg, -1)
end

# ╔═╡ 0179604e-7ccc-11eb-37bc-7df8411fb645
md"""
**Checkpoint**: Test your `inter_node_weighted_mistakes` function, run the following cell:
"""

# ╔═╡ 0163185c-7ccc-11eb-30dd-677a45cd9920
begin
  ex1_labels = Int[-1, -1, 1, 1, 1]
  ex1_data_weights = Float64[1., 2., .5, 1., 1.]

  @test inter_node_weighted_mistakes(ex1_labels, ex1_data_weights) == (2.5, -1)
  @test inter_node_weighted_mistakes(ex1_labels, ex1_data_weights)[2] == -1
end

# ╔═╡ 0149a662-7ccc-11eb-071c-0fb7dd06e885
md"""
Recall that the **classification error** is defined as follows:

$$\mbox{classification error} = \frac{\mbox{\# mistakes}}{\mbox{\# all data points}}$$

**Quiz Question:** If we set the weights $\mathbf{\alpha} = 1$ for all data points, how is the weight of mistakes $\mbox{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})$ related to the `classification error`?
"""

# ╔═╡ 0131e5be-7ccc-11eb-2890-7577dda5d4de
## Answer: N × classifcation Error, N num. of data points
begin
  nex_labels = Int[-1, -1, 1, 1, 1]
  nex_data_weights = Float64[1., 1., 1., 1., 1.]  ## All 1s
  inter_node_weighted_mistakes(nex_labels, nex_data_weights)
end

# ╔═╡ 074dee62-7ccd-11eb-3b98-bd0e15b4de11
md"""
#### Function to pick best feature to split on

We continue modifying our decision tree code from the earlier assignment to incorporate weighting of individual data points. The next step is to pick the best feature to split on.

The `best_splitting_feature` function is similar to the one from the earlier assignment with two minor modifications:

  1. The function `best_splitting_feature` should now accept an extra parameter `data_weights` to take account of weights of data points.

  2. Instead of computing the number of mistakes in the left and right side of the split, we compute the weight of mistakes for both sides, add up the two weights, and divide it by the total weight of the data.
"""

# ╔═╡ 0731dd4e-7ccd-11eb-12fa-4d42c67876b2
function best_splitting_feature(df::DF, features::Vector{Symbol},
                                target::Symbol, data_weights)

  best_feature, best_error = nothing, Base.Inf

  for f ∈ features
    l_split = df[df[!, f] .== 0, :]
    r_split = df[df[!, f] .== 1, :]

    ## Apply the same filtering to data_weights to create l_data_weights,
    ## r_data_weights
    l_data_weights = data_weights[df[!, f] .== 0, :]
    r_data_weights = data_weights[df[!, f] .== 1, :]

    ## Calculate the weight of mistakes for left and right sides
    l_w_mistakes, _ = inter_node_weighted_mistakes(l_split[!, target], l_data_weights)
    r_w_mistakes, _ = inter_node_weighted_mistakes(r_split[!, target], r_data_weights)

    # Compute weighted error
    s_1, s_2 = sum(l_data_weights), sum(r_data_weights)
    error = (l_w_mistakes + r_w_mistakes) / (s_1 + s_2)

    if error < best_error
      best_error = error
      best_feature = f
    end
  end

  best_feature
end

# ╔═╡ 0715ebb6-7ccd-11eb-20d9-87257af20d51
md"""
**Checkpoint**: Now, we have another checkpoint to make sure you are on the right track
"""

# ╔═╡ 06e27130-7ccd-11eb-0a51-851d7325ad47
begin
  tr_data_weights = fill(1.5, size(train_data)[1])

  @test string(best_splitting_feature(train_data, features, Target,
    tr_data_weights)) == "term_ 60 months"
end

# ╔═╡ 06b1c8aa-7ccd-11eb-08e3-635d3737c018
md"""
**Very Optional**. Relationship between weighted error and weight of mistakes

By definition, the weighted error is the weight of mistakes divided by the weight of all data points, so

$$\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}]}{\sum_{i=1}^{n} \alpha_i} = \frac{\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\sum_{i=1}^{n} \alpha_i}.$$

In the code above, we obtain $\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}})$ from the two weights of mistakes from both sides, $\mathrm{WM}(\mathbf{\alpha}_{\mathrm{left}}, \mathbf{\hat{y}}_{\mathrm{left}})$ and $\mathrm{WM}(\mathbf{\alpha}_{\mathrm{right}}, \mathbf{\hat{y}}_{\mathrm{right}})$. First, notice that the overall weight of mistakes $\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})$ can be broken into two weights of mistakes over either side of the split:

```math
\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})
= \sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}]
= \sum_{\mathrm{left}} \alpha_i \times 1[y_i \neq \hat{y_i}]
 + \sum_{\mathrm{right}} \alpha_i \times 1[y_i \neq \hat{y_i}]\\
= \mathrm{WM}(\mathbf{\alpha}_{\mathrm{left}}, \mathbf{\hat{y}}_{\mathrm{left}}) + \mathrm{WM}(\mathbf{\alpha}_{\mathrm{right}}, \mathbf{\hat{y}}_{\mathrm{right}})
```

We then divide through by the total weight of all data points to obtain $\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}})$:

```math
\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}})
= \frac{\mathrm{WM}(\mathbf{\alpha}_{\mathrm{left}}, \mathbf{\hat{y}}_{\mathrm{left}}) + \mathrm{WM}(\mathbf{\alpha}_{\mathrm{right}}, \mathbf{\hat{y}}_{\mathrm{right}})}{\sum_{i=1}^{n} \alpha_i}
```

"""

# ╔═╡ 0694786a-7ccd-11eb-0a69-8fbbd0859786
md"""
#### Building the tree

With the above functions implemented correctly, we are now ready to build our decision tree. Recall from the previous assignments that each node in the decision tree is represented as a dictionary which contains the following keys:

    {
       'is_leaf'            : True/False.
       'prediction'         : Prediction at the leaf node.
       'left'               : (dictionary corresponding to the left tree).
       'right'              : (dictionary corresponding to the right tree).
       'features_remaining' : List of features that are posible splits.
    }

Let us start with a function that creates a leaf node given a set of target values:
"""

# ╔═╡ 068154f6-7ccd-11eb-05df-8b16a1439d68
function create_leaf(target_values, data_weights;
                     splitting_feature=nothing, left=nothing,
                     right=nothing, is_leaf=true)

  ## Create a leaf node
  leaf = Dict{Symbol, Any}(
                           :is_leaf => is_leaf,
                           :splitting_feature => splitting_feature,
                           :left => left,
                           :right => right
                           )

  if is_leaf
    ## Compute weight of mistakes.
    _, best_class = inter_node_weighted_mistakes(target_values, data_weights)
    ## Store the predicted class (1 or -1) in leaf['prediction']
    leaf[:prediction] = best_class
  end
	
  leaf
end

# ╔═╡ 0100bf38-7ccc-11eb-34ff-9b3613ca036c
md"""
And now the function that learns a weighted decision tree recursively and implements 3 stopping conditions:

  1. All data points in a node are from the same class.
  2. No more features to split on.
  3. Stop growing the tree when the tree depth reaches max_depth.

"""

# ╔═╡ b6fc15ae-7a37-11eb-071c-0fb7dd06e885
function weighted_decision_tree_create(df::DF, features, target, data_weights; 				curr_depth=1, max_depth=10, verbose=false, ϵ=1e-15)

  rem_features = copy(features)
  target_vals = df[!, target]

  if verbose
    println("-------------------------------------------------------------")
    @printf("Subtree, depth = %s (%s data points).\n", curr_depth,
            size(target_vals)[1])
  end

  if inter_node_weighted_mistakes(target_vals, data_weights)[1] ≤ ϵ
    ## Stopping cond1: no mistakes at current node?
    verbose && @printf("Stopping condition 1 reached.")
    return create_leaf(target_vals, data_weights)

  elseif length(rem_features) == 0
    ## Stopping cond2: no remaining features to consider splitting on?
    verbose && @printf("Stopping condition 2 reached.")
    return create_leaf(target_vals, data_weights)

  elseif curr_depth > max_depth
    ## Additional stopping condition (limit tree depth)
    verbose && @printf("Early stopping condition 1 reached. Maximum depth.")
    return create_leaf(target_vals, data_weights)
  end

  ## Now Find the best splitting feature
  splitting_feature = best_splitting_feature(df, features, target, data_weights)

  # Split on the best feature that we found.
  l_split = df[df[!, splitting_feature] .== 0, :]
  r_split = df[df[!, splitting_feature] .== 1, :]
  l_dweights = data_weights[df[!, splitting_feature] .== 0, :]
  r_dweights = data_weights[df[!, splitting_feature] .== 1, :]

  deleteat!(rem_features,
            findall(x -> x == splitting_feature, rem_features))

  verbose && @printf("Split on feature %s. (%s, %s)\n", splitting_feature,
                     size(l_split)[1], size(r_split)[1])

  # Create a leaf node if the split is "perfect"
  if size(l_split)[1] == size(df)[1]
    verbose && println("Creating leaf node.")
    return create_leaf(l_split[!, target], data_weights)
  end

  if size(r_split)[1] == size(df)[1]
    verbose && println("Creating leaf node.")
    return create_leaf(r_split[!, target], data_weights)
  end

  # Recurse on left and right subtrees
  ltree = weighted_decision_tree_create(l_split, rem_features, target,
		l_dweights; curr_depth=curr_depth + 1, max_depth, verbose, ϵ)

  rtree = weighted_decision_tree_create(r_split, rem_features, target, 
		r_dweights; curr_depth=curr_depth + 1, max_depth, verbose, ϵ)

  return create_leaf(target_vals, data_weights;
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
  small_data_weights = ones(Float64, size(train_data)[1])

  small_dt = weighted_decision_tree_create(train_data, features, Target,      small_data_weights; max_depth=2, verbose=true)

  @test count_nodes(small_dt) == 7
end

# ╔═╡ dfb94004-7d62-11eb-071c-0fb7dd06e885
small_dt

# ╔═╡ c872733c-7b34-11eb-05df-8b16a1439d68
md"""
#### Making Predictions with a weighted decision tree

we will use our `classify` function (cf. include).


### Evaluating the tree

Now, we will write a function to evaluate a decision tree by computing the classification error of the tree on the given dataset.

Again, recall that the **classification error** is defined as follows:
$$\mbox{classification error} = \frac{\mbox{\# mistakes}}{\mbox{\# all data points}}$$

we will use our `eval_classification_error` function (cf. include).
"""

# ╔═╡ 75fc81fc-7d63-11eb-0a69-8fbbd0859786
round(eval_classification_error(small_dt, test_data, Target),
  digits=4)

# ╔═╡ 75da5262-7d63-11eb-05df-8b16a1439d68
md"""
#### Example: Training a weighted decision tree

To build intuition on how weighted data points affect the tree being built, consider the following:

Suppose we only care about making good predictions for the **first 10 and last 10 items** in `train_data`, we assign weights:
  - 1 to the last 10 items
  - 1 to the first 10 items
  - and 0 to the rest.

Let us fit a weighted decision tree with `max_depth = 2`.
"""

# ╔═╡ 75c205c2-7d63-11eb-37bc-7df8411fb645
begin
  	## Assign weights
  	ex_data_weights = Float64[ones(Float64, 10)..., 
		zeros(Float64, size(train_data)[1] - 20)..., ones(Float64, 10)...]

	## Train a weighted decision tree model.
	small_data_dt_subset20 = weighted_decision_tree_create(train_data, features,
		Target, ex_data_weights, max_depth=2)
end

# ╔═╡ 75a67960-7d63-11eb-30dd-677a45cd9920
md"""
Now, we will compute the classification error on the subset20, i.e. the subset of data points whose weight is 1 (namely the first and last 10 data points).
"""

# ╔═╡ a56f646c-7d64-11eb-071c-0fb7dd06e885
begin
  subset20 = vcat(first(train_data, 10), last(train_data, 10))
  round(eval_classification_error(small_data_dt_subset20, subset20, Target),
    digits=4)
end

# ╔═╡ a5394d28-7d64-11eb-34ff-9b3613ca036c
md"""
Now, let us compare the classification error of the model `small_data_dt_subset20` on the entire train\_data set:
"""

# ╔═╡ a5213ba2-7d64-11eb-1d64-d7e7ff02159c
round(eval_classification_error(small_data_dt_subset20, train_data, Target),
  digits=4)

# ╔═╡ 1892ec06-7d66-11eb-37bc-7df8411fb645
md"""
The model `small_data_dt_subset20` performs **a lot** better on `subset20` than on `train_data`.

So, what does this mean?
* The points with higher weights are the ones that are more important during the training process of the weighted decision tree.
* The points with zero weights are basically ignored during training.

**Quiz Question**: Will you get the same model as `small_data_dt_subset20` if you trained a decision tree with only the 20 data points with non-zero weights from the set of points in `subset_20`?
  - Yes
"""

# ╔═╡ a57ffa0a-7d66-11eb-05df-8b16a1439d68
md"""
### Implementing our own Adaboost (on decision stumps)


Now that we have a weighted decision tree working, it takes only a bit of work to implement Adaboost. For the sake of simplicity, let us stick with **decision tree stumps** by training trees with **`max_depth=1`**.

Recall from the lecture the procedure for Adaboost:

1\. Start with unweighted data with $\alpha_j = 1$

2\. For t = 1,...T:
  * Learn $f_t(x)$ with data weights $\alpha_j$
  * Compute coefficient $\hat{w}_t$:
     $$\hat{w}_t = \frac{1}{2}\ln{\left(\frac{1- \mbox{E}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\mbox{E}(\mathbf{\alpha}, \mathbf{\hat{y}})}\right)}$$

  * Re-compute weights $\alpha_j$:
     $$\alpha_j \gets \begin{cases}
     \alpha_j \exp{(-\hat{w}_t)} & \text{ if }f_t(x_j) = y_j\\
     \alpha_j \exp{(\hat{w}_t)} & \text{ if }f_t(x_j) \neq y_j
     \end{cases}$$

  * Normalize weights $\alpha_j$:
      $$\alpha_j \gets \frac{\alpha_j}{\sum_{i=1}^{N}{\alpha_i}}$$

Write the function `adaboost_with_tree_stumps` (given Python skeleton).

"""

# ╔═╡ f962b6f0-7d66-11eb-0a69-8fbbd0859786
function adaboost_with_tree_stumps(df, features, target;
                                   num_tree_stumps=2, verbose=false)
  α = ones(Float64, size(df)[1])  ## start with unweighted data
  weights, tree_stumps = Float64[], []
  target_vals = select(df, target)[:, target]

  for _t ∈ 1:num_tree_stumps
    if verbose
      println("=====================================================")
      @printf("Adaboost Iteration %d\n", _t)
      println("=====================================================")
    end

    ## Learn a weighted decision tree stump. Use max_depth=1
    tree_stump = weighted_decision_tree_create(df, features, target, α;
                                               max_depth=1, verbose=verbose)
    push!(tree_stumps, tree_stump)
    preds = map(x -> classify(tree_stump, x), eachrow(df)) 

    ## Boolean array indicating whether each data point was correctly classified
    is_correct = preds .== target_vals
    is_wrong   = preds .≠ target_vals

    weighted_err = sum(α[is_wrong]) / sum(α)  ## Compute weighted error

    ## Compute model coefficient using weighted error
    weight = .5 * log((1 - weighted_err) / weighted_err)
    push!(weights, weight)

    ## Adjust weights on data point
    adjustment = ifelse.(is_correct, exp(-weight), exp(weight))

    ## Scale α by: × by adjustment, then normalize data points weights
    α = α .* adjustment
    α /= sum(α)
  end
  weights, tree_stumps
end

# ╔═╡ a55ef7c4-7d66-11eb-37bc-7df8411fb645
md"""
### Checking our Adaboost code

Train an ensemble of **two** tree stumps and see which features those stumps split on. We will run the algorithm with the following parameters:
  - `train_data`
  - `features`
  - `target`
  - `num_tree_stumps = 2`
"""

# ╔═╡ a516c486-7d66-11eb-30dd-677a45cd9920
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, Target;	num_tree_stumps=2, verbose=true)

# ╔═╡ a4f7d690-7d66-11eb-071c-0fb7dd06e885
function  print_stump(tree, name=:root)
    split_name = tree[:splitting_feature] # ex. 'term. 36 months'

    if isnothing(split_name)
        print("(leaf, label: $(tree[:prediction]))")
        return nothing
  end
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

# ╔═╡ a4dd0c0a-7d66-11eb-2890-7577dda5d4de
with_terminal() do
  print_stump(tree_stumps[1])
end

# ╔═╡ a4c39572-7d66-11eb-34ff-9b3613ca036c
with_terminal() do
  print_stump(tree_stumps[2])
end

# ╔═╡ a4aa3a96-7d66-11eb-1d64-d7e7ff02159c
md"""
If our Adaboost is correctly implemented, the following things should be true:

  - `tree_stumps[1]` should split on **term. 60 months** with the prediction -1 on the left and +1 on the right.
  - `tree_stumps[2]` should split on **grade.A** with the prediction -1 on the left and +1 on the right.
  - Weights should be approximately `[0.151, 0.187]`

**Reminders**
  - Stump weights ($\mathbf{\hat{w}}$) and data point weights ($\mathbf{\alpha}$) are two different concepts.
  - Stump weights ($\mathbf{\hat{w}}$) tell you how important each stump is while making predictions with the entire boosted ensemble.
  - Data point weights ($\mathbf{\alpha}$) tell you how important each data point is while training a decision stump.
"""

# ╔═╡ 28bccc2c-7d80-11eb-2c80-6b7e9646f76e
md"""
### Training a boosted ensemble of 10 stumps

Let us train an ensemble of 10 decision tree stumps with Adaboost. We run the `adaboost_with_tree_stumps` function with the following parameters:
  - `train_data`
  - `features`
  - `target`
  - `num_tree_stumps = 10`
"""

# ╔═╡ 4653cf88-7d80-11eb-354e-f54c38b7781c
stump_weights10, tree_stumps10 = adaboost_with_tree_stumps(train_data, features, Target,
                                                           num_tree_stumps=10)

# ╔═╡ 28a27e62-7d80-11eb-2d86-2ff7694e596e
md"""
#### Making predictions

Recall from the lecture that in order to make predictions, we use the following formula:

$$\hat{y} = sign\left(\sum_{t=1}^T \hat{w}_t f_t(x)\right)$$

We need to do the following things:
- Compute the predictions $f_t(x)$ using the $t$-th decision tree
- Compute $\hat{w}_t f_t(x)$ by multiplying the `stump_weights` with the predictions $f_t(x)$ from the decision trees
- Sum the weighted predictions over each stump in the ensemble.

Complete the following skeleton for making predictions:
"""

# ╔═╡ 2885d134-7d80-11eb-3039-85b1d81f2b21
function predict_adaboost(stump_weights, tree_stumps, df)
	
  scores = zeros(Float64, size(df)[1])

  for (ix, tree_stump) ∈ enumerate(tree_stumps)
	preds = map(x -> classify(tree_stump, x), eachrow(df))
    scores += stump_weights[ix] .* preds
  end
	
  ifelse.(scores .> zero(eltype(scores)), 1, -1)
end

# ╔═╡ 69dc69d2-7d81-11eb-2afd-216aca94000f
begin
  preds10 = predict_adaboost(stump_weights, tree_stumps, test_data)
  acc10 = sum(test_data[!, Target] .== preds10) / size(test_data)[1]

  with_terminal() do
    @printf("\nAccuracy of 10-component ensemble = %1.4f\n", acc10)
  end
end

# ╔═╡ 4205b9e0-7d81-11eb-283e-174f221087c0
stump_weights10

# ╔═╡ 525dffa0-7d81-11eb-346f-a78d0b34253e
md"""
**Quiz Question:** Are the weights monotonically decreasing, monotonically increasing, or neither?
  - Neither (from iter1 to iter 2: increase, after that decrease)

**Reminder**: Stump weights ($\mathbf{\hat{w}}$) tell you how important each stump is while making predictions with the entire boosted ensemble.
"""

# ╔═╡ 4214e72a-7d82-11eb-1d64-d7e7ff02159c
md"""
#### Performance plots

In this section, we will try to reproduce some of the performance plots dicussed in the lecture.

### How does accuracy change with adding stumps to the ensemble?

We will now train an ensemble with:
  - `train_data`
  - `features`
  - `target`
  - `num_tree_stumps = 30`

Once we are done with this, we will then do the following:
  - Compute the classification error at the end of each iteration.
  - Plot a curve of classification error vs iteration.

First, let's train the model.
"""

# ╔═╡ 62a40a70-7d82-11eb-34ff-9b3613ca036c
stump_weights30, tree_stumps30 = adaboost_with_tree_stumps(train_data, features, Target, num_tree_stumps=30)

# ╔═╡ 72b17ef2-7d82-11eb-30dd-677a45cd9920
md"""
##### Computing training error at the end of each iteration

Now, we will compute the classification error on the `train_data` and see how it is reduced as trees are added.
"""

# ╔═╡ 728ff304-7d82-11eb-071c-0fb7dd06e885
function calc_error(df, label, stump_weights, tree_stumps; 
		target=Target, n=30)
    err_all = Float64[]

    for ix ∈ 1:n
      preds = predict_adaboost(stump_weights[1:ix], tree_stumps[1:ix], df)
      err = 1.0 - sum(df[!, target] .== preds) / size(df)[1]
      push!(err_all, err)
      @printf("Iteration %2d, %s error = %2.5f\n", ix, label, err_all[ix])
  end

  err_all
end

# ╔═╡ 72750690-7d82-11eb-2890-7577dda5d4de
error_all = calc_error(train_data, :training, stump_weights30, tree_stumps30)

# ╔═╡ 56fc7b38-7d84-11eb-05df-8b16a1439d68
test_error_all = calc_error(test_data, :test, stump_weights30, tree_stumps30)

# ╔═╡ 05a54422-7d84-11eb-37bc-7df8411fb645
begin
  plot(collect(1:30), error_all, linewidth=3.0, label="Training error")
  plot!(collect(1:30), test_error_all, linewidth=3.0, label="Testing error")
end

# ╔═╡ Cell order:
# ╟─5435ff66-7a2b-11eb-1474-b96dcad21315
# ╠═77311686-7a2b-11eb-2bc7-95b2a2211969
# ╠═81b7a9e0-7cc8-11eb-1d64-d7e7ff02159c
# ╠═e9f8e942-7b29-11eb-1d64-d7e7ff02159c
# ╟─771130dc-7a2b-11eb-170f-934c11fceede
# ╠═76f62e40-7a2b-11eb-095e-f36e207f06a2
# ╟─76de16d4-7a2b-11eb-09c6-e36b6c893c1f
# ╠═76c17c4a-7a2b-11eb-2c47-97106bd58f99
# ╟─76a83684-7a2b-11eb-1961-cb133db8c164
# ╠═768d6b08-7a2b-11eb-388a-8763c8da0a51
# ╟─7673d8e4-7a2b-11eb-02b4-2f837181b4e9
# ╠═3773355e-7cc8-11eb-2890-7577dda5d4de
# ╠═542db682-7a2e-11eb-35ad-25c82d6919e0
# ╠═6327985a-7cc8-11eb-071c-0fb7dd06e885
# ╠═7641d404-7a2b-11eb-042d-bbc2c500ac59
# ╟─760c171a-7a2b-11eb-1341-19ba5a1824a1
# ╠═75c1a64e-7a2b-11eb-21f6-55dd3a8ce164
# ╠═75a52c44-7a2b-11eb-2f02-2312f3a3dac9
# ╠═75736a9c-7a2b-11eb-2278-ab37447cf22d
# ╠═116eb5f8-7a33-11eb-2c4c-5f937540f987
# ╟─a5c56d68-7a32-11eb-2ef6-e58ed65dc561
# ╠═a5a1cdf6-7a32-11eb-18dd-0dfee8eb885c
# ╟─a56e44e0-7a32-11eb-2587-6f5f8b8c21f3
# ╠═6fad8afe-7a33-11eb-3208-6b289c3ccd23
# ╟─6f88e4f6-7a33-11eb-10cf-e3e8608e616b
# ╠═49f07e5c-7b2b-11eb-34ff-9b3613ca036c
# ╟─0179604e-7ccc-11eb-37bc-7df8411fb645
# ╠═0163185c-7ccc-11eb-30dd-677a45cd9920
# ╟─0149a662-7ccc-11eb-071c-0fb7dd06e885
# ╠═0131e5be-7ccc-11eb-2890-7577dda5d4de
# ╟─074dee62-7ccd-11eb-3b98-bd0e15b4de11
# ╠═0731dd4e-7ccd-11eb-12fa-4d42c67876b2
# ╟─0715ebb6-7ccd-11eb-20d9-87257af20d51
# ╠═06e27130-7ccd-11eb-0a51-851d7325ad47
# ╟─06b1c8aa-7ccd-11eb-08e3-635d3737c018
# ╟─0694786a-7ccd-11eb-0a69-8fbbd0859786
# ╠═068154f6-7ccd-11eb-05df-8b16a1439d68
# ╟─0100bf38-7ccc-11eb-34ff-9b3613ca036c
# ╠═b6fc15ae-7a37-11eb-071c-0fb7dd06e885
# ╟─cdb69d9c-7b2b-11eb-12fa-4d42c67876b2
# ╠═cd9e2b4a-7b2b-11eb-20d9-87257af20d51
# ╠═dfb94004-7d62-11eb-071c-0fb7dd06e885
# ╟─c872733c-7b34-11eb-05df-8b16a1439d68
# ╠═75fc81fc-7d63-11eb-0a69-8fbbd0859786
# ╟─75da5262-7d63-11eb-05df-8b16a1439d68
# ╠═75c205c2-7d63-11eb-37bc-7df8411fb645
# ╟─75a67960-7d63-11eb-30dd-677a45cd9920
# ╠═a56f646c-7d64-11eb-071c-0fb7dd06e885
# ╟─a5394d28-7d64-11eb-34ff-9b3613ca036c
# ╠═a5213ba2-7d64-11eb-1d64-d7e7ff02159c
# ╟─1892ec06-7d66-11eb-37bc-7df8411fb645
# ╟─a57ffa0a-7d66-11eb-05df-8b16a1439d68
# ╠═f962b6f0-7d66-11eb-0a69-8fbbd0859786
# ╟─a55ef7c4-7d66-11eb-37bc-7df8411fb645
# ╠═a516c486-7d66-11eb-30dd-677a45cd9920
# ╠═a4f7d690-7d66-11eb-071c-0fb7dd06e885
# ╠═a4dd0c0a-7d66-11eb-2890-7577dda5d4de
# ╠═a4c39572-7d66-11eb-34ff-9b3613ca036c
# ╟─a4aa3a96-7d66-11eb-1d64-d7e7ff02159c
# ╟─28bccc2c-7d80-11eb-2c80-6b7e9646f76e
# ╠═4653cf88-7d80-11eb-354e-f54c38b7781c
# ╟─28a27e62-7d80-11eb-2d86-2ff7694e596e
# ╠═2885d134-7d80-11eb-3039-85b1d81f2b21
# ╠═69dc69d2-7d81-11eb-2afd-216aca94000f
# ╠═4205b9e0-7d81-11eb-283e-174f221087c0
# ╟─525dffa0-7d81-11eb-346f-a78d0b34253e
# ╟─4214e72a-7d82-11eb-1d64-d7e7ff02159c
# ╠═62a40a70-7d82-11eb-34ff-9b3613ca036c
# ╟─72b17ef2-7d82-11eb-30dd-677a45cd9920
# ╠═728ff304-7d82-11eb-071c-0fb7dd06e885
# ╠═72750690-7d82-11eb-2890-7577dda5d4de
# ╠═56fc7b38-7d84-11eb-05df-8b16a1439d68
# ╠═05a54422-7d84-11eb-37bc-7df8411fb645
