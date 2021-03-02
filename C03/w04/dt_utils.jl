using Printf


function inter_node_num_mistakes(labels_in_node::AbstractVector{T}) where {T <: Real}
    length(labels_in_node) == 0 && return 0  ## Corner case

  n = length(labels_in_node)
  n_pos = sum(labels_in_node .== 1)  ## Count the number of 1's (safe loans)
    n_neg = n - n_pos ## or(labels_in_node == -1).sum()

  ## Return the number of mistakes that the majority classifier makes.
  return n_pos ≥ n_neg ? n_neg : n_pos
end


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


function create_leaf(target_values;
                     splitting_feature=nothing, left=nothing, right=nothing, is_leaf=true)
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

  return leaf
end


function count_nodes(tree)
    isnothing(tree) || tree[:is_leaf] ?
    1 : 1 + count_nodes(tree[:left]) + count_nodes(tree[:right])
end


function count_leaves(tree)
    isnothing(tree) || tree[:is_leaf] ?
    1 : count_nodes(tree[:left]) + count_nodes(tree[:right])
end


function classify(tree, x; annotate=false)
  # if the node is a leaf node.
  if tree[:is_leaf]
    annotate && @printf("At leaf, predicting %s\n", tree[:prediction])
    return tree[:prediction]
  end

  ## split on feature.
  split_feature_value = x[tree[:splitting_feature]]
  annotate &&
  @printf("Split on %s => %s\n", tree[:splitting_feature], split_feature_value)

  split_feature_value == 0 ? classify(tree[:left], x; annotate) :
  classify(tree[:right], x; annotate)
end


function eval_classification_error(tree, df::DF,
                                   target::Symbol) where {DF <: Union{DataFrame, SubDataFrame}}
  ## 1. get the predictions
  ŷ = map(x -> classify(tree, x), eachrow(df))

  ## 2. calculate the classification error
  num_mistakes = sum(ŷ .≠ df[!, target])
  return num_mistakes / size(df)[1]
end
