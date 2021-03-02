using SparseArrays
using DataFrames

#
# Reading file as a strings
const Stop_Words = open("./stopwords-en.txt", "r") do file
  read(file, String) |>
    s -> split(s, "\n") |>
    a -> Dict{String, Bool}(w => true for w in a)
end

const P_REXPR1 = r"[\.,:;\"'\-?!\+\=_\*#%(){}\[\]$&~/0-9\|@><\\\^`]"

remove_punctuation(text::AbstractString) = replace(text, P_REXPR1 => " ") |>
  strip |> string

remove_punctuation(vtext::Vector{String}) = remove_punctuation.(vtext)

cleanup(text::AbstractString) = replace(text, P_REXPR1 => " ") |>
  strip |> string |> s -> lowercase.(s)

cleanup(vtext::Vector{String}) = cleanup.(vtext)

function word_count(text::String)
    hwc = Dict{String, Int}()
    for w ∈ split(text)
      w = lowercase(w)
      hwc[w] = get(hwc, w, 0) + 1
    end
    hwc
  end

## no need to worry about Vector{Union{Missing, String}}...
word_count(vtext::Vector{String}) = word_count.(vtext)

function gen_vocab(text::String; trim_len=3, trim_count=2)
  lowercase.(split(text)) |>
    a_ -> filter(w -> !haskey(Stop_Words, w), a_) |>
    vtext -> reduce(vcat, vtext; init=String[]) |>
    unique
end

function gen_token_ix(vtext::Vector{String})
  reduce(vcat, vtext) |>
     unique |>
     words -> filter(w -> length(w) > 2, words) |>
     vocab -> Dict{String, Int}(w => ix for (ix, w) in enumerate(vocab))
end

function h_encode(text::String; token_ix=token_ix)
  @assert !isnothing(token_ix)
  v = spzeros(Int8, length(token_ix)) #
  # v = zeros(Int8, length(token_ix))

  for w ∈ split(text)
    ix = get(token_ix, lowercase(w), -1)
    ix == -1 && continue
    v[ix] += one(Int8)
  end

  v
end

h_encode(vtext::Vector{String}; token_ix=token_ix) = h_encode.(vtext; token_ix)

function get_data(df::DF, features, output) where {DF <: Union{DataFrame, SubDataFrame}}
  s_features = [Symbol(f) for f ∈ features]
  df[:, :intercept] .= 1.0
  s_features = [:intercept, s_features...]
  X_matrix = convert(Matrix, select(df, s_features))
  y = df[!, output]

  (X_matrix, y)
end
