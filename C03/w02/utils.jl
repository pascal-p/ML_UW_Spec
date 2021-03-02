using Random
using MLJ
using DataFrames
using LinearAlgebra


"""
Split a given Dataframe into 2 sub-dataframe allowing
for training/testing dataset split
"""
function train_test_split(df::DF; split=0.8, seed=42, shuffled=true) where {DF <: Union{DataFrame, SubDataFrame}}
  Random.seed!(seed)
  (nr, nc) = size(df)
  nrp = round(Int, nr * split)
  row_ixes = shuffled ? shuffle(1:nr) : collect(1:nr)

  df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
  df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)

  (df_train, df_test)
end

"""
From a Dataframe to matrix
from week2
"""
function get_data(df::DF, features, output;
                  colname=:constant) where {DF <: Union{DataFrame, SubDataFrame}}
  df[:, colname] .= 1.0
  features = [colname, features...]
  X_matrix = convert(Matrix, select(df, features)) # to get a matrix
  y = df[!, output]                                # => to get a vector
  return (X_matrix, y)
end


"""
Assume feature_matrix is a matrix containing the features as columns
and weights is a corresponding array

from week2
"""
function predict_output(X::Matrix{T}, weights::Vector{T}) where {T <: Real}
    X * weights
end

"""
Calculate RSS given weights vector

depends on predict_output
from week2

y::SubArray{Float64,1, Array{Float64,1}, Tuple{UnitRange{Int64}}, true}
"""
function calc_rss(X::Matrix{T}, y, weights::Vector{T}) where {T <: Real}
  preds = predict_output(X, weights)
  sum((preds .- y) .^ 2)
end


"""
Calculate RSS given a (MLJ) machine which is a wrapper combining a model
and its data.

depends on MLJ predict()

mach::MLJBase.Machine{MLJLinearModels.RidgeRegressor,true}
"""
function get_rss(mach::Machine, X::DF, y::Vector{T}) where {DF <: Union{DataFrame, SubDataFrame}, T <: Real}
  ŷ = predict(mach, X)     # First get the predictions
  diff = y .- ŷ            # Then compute the residuals/errors
  rss = sum(diff .* diff)  # Then square and add them up
  return rss
end

function normalize_features(X::Matrix)
  n = size(X)[2]
  norms = zeros(eltype(X), n)'
  for ix ∈ 1:n
    norms[ix] = norm(X[:, ix])
  end
  (X ./ norms, norms)
end


"""
Select a random sample percentage of an index range
"""
function sampling(range::Integer, perc::Float64; seed=42)
  up_lim = ceil(Integer, perc * range)
  randperm(MersenneTwister(seed), range)[1:up_lim]
end


"""
Given a DF and the features to encode, do on hot encoding of each
feature

on-place transformation
"""
function hot_encode!(df::DF;
                     features=Features) where {DF <: Union{DataFrame, SubDataFrame}}
  for f ∈ features
    ## get unique value over column feature
    fval = reduce(vcat, df[!, f]) |> unique

    ## apply hotencoding inplace
    ## cf. https://stackoverflow.com/questions/64565276/julia-dataframes-how-to-do-one-hot-encoding
    transform!(df,
      f .=> [ByRow(v -> occursin(x, v) ? 1 : 0) for x ∈ fval] .=> Symbol.(f, fval))

    ## get rid of initial feature (column)
    select!(df, Not(f))
  end
end


"""
Condition the data, adjusting to right MLG scientific type

features, target of type F_ST (Feature SciType)
"""
function data_prep_Xy(df::DF, features, target) where {DF <: Union{DataFrame, SubDataFrame}}

  y_target = target.names[1]

  X = select(df, features.names)
  coerce!(X, features.st_map...)

  y = select(df, y_target)
  coerce!(y, target.st_map...)
  y = y[:, y_target]   ## cast to vector

  return (X, y)
end

function data_prep(df::DF, features) where {DF <: Union{DataFrame, SubDataFrame}}
  coerce(df, features.st_map...)
end
