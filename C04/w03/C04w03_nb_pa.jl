### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 2a641766-809f-11eb-274a-8fdb234c0d08
begin
  using Pkg
  Pkg.activate("MLJ_env", shared=true)

  using CSV
  using DataFrames
  using PlutoUI
  using Test
  using Printf
  using Plots
  using LinearAlgebra
  using SparseArrays
  using TextAnalysis
  using Random
  using Distances
  using Statistics
  using Dates
end

# ╔═╡ f701bf4a-809e-11eb-2f1f-bb7dad75bfcc
md"""
## k-means with text data

In this assignment we will
  - Cluster Wikipedia documents using k-means
  - Explore the role of random initialization on the quality of the clustering
  - Explore how results differ after changing the number of clusters
  - Evaluate clustering, both quantitatively and qualitatively

When properly executed, clustering uncovers valuable insights from a set of unlabeled documents.
"""

# ╔═╡ 692a39c8-80a4-11eb-2077-e9f391787416
# Pkg.add("TextAnalysis")
# Pkg.add("Distances")

# ╔═╡ 5b3b148e-809f-11eb-3631-cf4682cbd959
begin
  """compute norm of a sparse vector"""
  norm_sv(v::AbstractVector) = sqrt.(dot(v, v'))

  function norm_sv(m::AbstractMatrix)
    n = size(m)[1]
    reshape([norm_sv(m[ix, :]) for ix in 1:n], n, 1)
  end
end

# ╔═╡ 97d6cb18-809f-11eb-075a-7b3c5354ce4c
## Norm of a sparse vector
begin
  v = sparsevec([1 2 0 0 0 3 0 4 0])
  nv = norm_sv(v)

  @test issparse(v)
  @test nv ≈ √(1^2 + 2^2 + 3^2 + 4^2)
end

# ╔═╡ 97d2bd70-8239-11eb-2a8c-efea2584e6d6
## Norm of a dense Matrix (row only)
begin
  m = [1 2 3; 4 1 2; 1 3 3; 1 0 2]  # 4×3
  dim = size(m)[1]

  norm_sv(m)

  ## then normalize matrix
  norm_m = m ./ norm_sv(m)

  ## test all rows have norm 1
  @test all(v -> norm_sv(v) ≈ 1., eachrow(norm_m))
end

# ╔═╡ 54ce09fa-8246-11eb-35fc-a9ba218d8c76
## Norm of a sparse Matrix (row only)
begin
  mₛ = sparse([1 2 0 3; 4 0 0 0; 0 0 3 1; 1 0 2 0])  # 4×4
  dimₛ = size(mₛ)[1]

  norm_sv(mₛ)

  ## then normalize matrix
  norm_mₛ = mₛ ./ norm_sv(mₛ)

  ## test all rows have norm 1
  @test all(v -> norm_sv(v) ≈ 1., eachrow(norm_mₛ))
end

# ╔═╡ 6ada72c6-80a0-11eb-2346-c3554265a4de
md"""
#### Load in the Wikipedia dataset

**Just a small subset of the data to validate our implementation.**

**Later we will scale and see...**
"""

# ╔═╡ 9ee529a8-80a0-11eb-23cc-b78ead6503e4
begin
  # just a subset
  wiki = train = CSV.File("../../ML_UW_Spec/data/people_wiki_500samples.csv";
                          header=true) |> DataFrame;
  size(wiki)
end

# ╔═╡ 16431f28-80a4-11eb-12a2-f3104b93f673
describe(wiki, :eltype, :nmissing, :first => first)

# ╔═╡ e98e9a5e-80a3-11eb-1d38-db4ff0f9575b
md"""
Let us add a unique id to each row of the wiki DataFrame
"""

# ╔═╡ fbebf142-80a3-11eb-0913-4b43d543a681
insertcols!(wiki, 1, :id => collect(1:size(wiki)[1]));

# ╔═╡ 637c61ac-80a4-11eb-21be-f1661d34b568
names(wiki)

# ╔═╡ a30ee624-81e8-11eb-0c75-a5715165ff36
md"""
#### Extract features
"""

# ╔═╡ 4543b590-80a5-11eb-3215-b9bb37e1e766
begin
  crps = Corpus(StringDocument.(wiki.text))
  update_lexicon!(crps)
end

# ╔═╡ fd36c16e-80a6-11eb-3a80-5d7b8cb48b38
dtm = DocumentTermMatrix(crps);

# ╔═╡ 4e94e7a2-80a7-11eb-0fb3-1955f645d0ca
ya_tf_idf = tf_idf(dtm);

# ╔═╡ 107cca2e-80ad-11eb-19b8-bf015a43b184
lexicon_size(crps)

# ╔═╡ 69c34f00-80ac-11eb-0b03-0362d6f5e84e
crps

# ╔═╡ b0d25bf0-80ae-11eb-00de-d57e0d22a000
dtm[1, :], ya_tf_idf[1, :]

# ╔═╡ 07757e58-8247-11eb-299a-97cbb8ce1460
md"""
As discussed in the previous assignment, Euclidean distance can be a poor metric of similarity between documents, as it unfairly penalizes long articles. For a reasonable assessment of similarity, we should disregard the length information and use length-agnostic metrics, such as cosine distance.

The k-means algorithm does not directly work with cosine distance, so we take an alternative route to remove length information: we normalize all vectors to be unit length. It turns out that Euclidean distance closely mimics cosine distance when all vectors are unit length. In particular, the squared Euclidean distance between any two vectors of length one is directly proportional to their cosine distance.

We can prove this as follows. Let $\mathbf{x}$ and $\mathbf{y}$ be normalized vectors, i.e. unit vectors, so that $\|\mathbf{x}\|=\|\mathbf{y}\|=1$. Write the squared Euclidean distance as the dot product of $(\mathbf{x} - \mathbf{y})$ to itself:

```math
\begin{align*}
\|\mathbf{x} - \mathbf{y}\|^2 &= (\mathbf{x} - \mathbf{y})^T(\mathbf{x} - \mathbf{y})\\
                              &= (\mathbf{x}^T \mathbf{x}) - 2(\mathbf{x}^T \mathbf{y}) + (\mathbf{y}^T \mathbf{y})\\
                              &= \|\mathbf{x}\|^2 - 2(\mathbf{x}^T \mathbf{y}) + \|\mathbf{y}\|^2\\
                              &= 2 - 2(\mathbf{x}^T \mathbf{y})\\
                              &= 2(1 - (\mathbf{x}^T \mathbf{y}))\\
                              &= 2\left(1 - \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}\right)\\
                              &= 2\left[\text{cosine distance}\right]
\end{align*}
```

This tells us that two **unit vectors** that are close in Euclidean distance are also close in cosine distance. Thus, the k-means algorithm (which naturally uses Euclidean distances) on normalized vectors will produce the same results as clustering using cosine distance as a distance metric.

"""

# ╔═╡ 99a6e9c4-8238-11eb-3acc-63a515a49b64
norm_ya_tf_idf = ya_tf_idf ./ norm_sv(ya_tf_idf);

# ╔═╡ 3f2f94e4-80af-11eb-32d2-a5e101797f53
size(ya_tf_idf), lexicon(crps)

# ╔═╡ fcfefb98-81e7-11eb-014a-b72b0477acab
# FIXME: disabled for now
# begin
#   update_inverse_index!(crps);
#   map_index_to_word = inverse_index(crps)
# end

# ╔═╡ 6c7e5bb4-823b-11eb-3833-254b4f5dafa7
## all rows are unit vector (vector with unit norm)
@test all(v -> norm_sv(v) ≈ 1., eachrow(norm_ya_tf_idf))

# @test all(ix -> abs(norm_sv(norm_ya_tf_idf[ix, :]) - 1.) ≤ 1e-8, collect(size(ya_tf_idf)[1]))

# ╔═╡ aaea9810-81e9-11eb-0452-83d979dca4c3
md"""
### Implement k-means

Let us implement the k-means algorithm. First, we choose an initial set of centroids. A common practice is to choose randomly from the data points.

Note: In practice, we highly recommend to use different seeds every time (for instance, by using the current timestamp).
"""

# ╔═╡ aad0780c-81e9-11eb-2344-63996006b0cb
function get_init_centroids(df, k; seed=42)
  """
  Randomly choose k data points as initial centroids
  """
    rng = MersenneTwister(seed);
    n = size(df)[1]
    rand_ix = randperm(rng, n)[1:k]  ## Pick k indices from range [1, n].

    ## Keep centroids as dense format, as many entries will be nonzero
    ## due to averaging.
    ## As long as at least one document in a cluster contains a word,
    ## it will carry a nonzero weight in the TF-IDF vector of the centroid.
    Matrix(df[rand_ix, :])
end

# ╔═╡ 446d44fa-81eb-11eb-0daa-83db97d8eec9
md"""
After initialization, the k-means algorithm iterates between the following two steps:

  1. Assign each data point to the closest centroid.
$$z_i \gets \mathrm{argmin}_j \|\mu_j - \mathbf{x}_i\|^2$$

  2. Revise centroids as the mean of the assigned data points.
$$\mu_j \gets \frac{1}{n_j}\sum_{i:z_i=j} \mathbf{x}_i$$

In pseudocode, we iteratively do the following:
```
cluster_assignment = assign_clusters(data, centroids)
centroids = revise_centroids(data, k, cluster_assignment)
```
"""

# ╔═╡ f1494764-81eb-11eb-02e8-e9a44cbbd905
md"""
#### Assigning clusters

How do we implement Step 1 of the main k-means loop above?

For the sake of demonstration, let's look at documents 100 through 102 as query documents and compute the distances between each of these documents and every other document in the corpus.

In the k-means algorithm, we will have to compute pairwise distances between the set of centroids and the set of documents.

"""

# ╔═╡ 444fc56a-81eb-11eb-27d2-1b249cae06d9
begin
    queries = norm_ya_tf_idf[100:102, :] ## Get the TF-IDF vectors for documents 100 through 102.
    ## Compute pairwise distances from every data point to each query vector.
    dist = pairwise(Euclidean(), norm_ya_tf_idf, queries, dims=1)
    @assert size(dist) == (size(norm_ya_tf_idf)[1], size(queries)[1])
end

# ╔═╡ 9716ee8a-81fb-11eb-3279-c19f1a5736a2
closest_cluster = argmin(dist; dims=2)

# ╔═╡ 95d6fcdc-81ff-11eb-0d5b-5b161dba559f
@test length(closest_cluster) == size(dist)[1]

# ╔═╡ 68962f52-8215-11eb-1cf2-1763b3ea2fcf
const ϵ = 1e-9 # tolerance for distances

# ╔═╡ aab3edea-81e9-11eb-2104-f18b12260e09
function assign_clusters(data, centroids)
    ## Compute distances between each data point and the set of centroids:
    dists_from_centroids = pairwise(Euclidean(ϵ), data, centroids, dims=1)

    ## Compute cluster assignments for each data point:
    argmin(dists_from_centroids; dims=2)
end

# ╔═╡ aa994aa0-81e9-11eb-1fca-f1b45c0220f0
md"""
**Checkpoint.** Let us check if Step 1 was implemented correctly.

With rows 1, 3, 5, and 7 of corpus as an initial set of centroids, we assign cluster labels to rows 30, 37, ..., and 99 of corpus.
"""

# ╔═╡ 8f71f718-8200-11eb-2dc1-4bc972d149f2
begin
  sample_data = norm_ya_tf_idf[50:4:99, :]
  sample_centroids = norm_ya_tf_idf[31:3:39, :]

  @assert size(sample_centroids)[1] == 3

  sample_cluster_assign = assign_clusters(sample_data, sample_centroids)
end

# ╔═╡ 20c45328-8237-11eb-1cf3-a9e5a4ee496f
size(sample_centroids), size(sample_data)

# ╔═╡ f812155c-8232-11eb-0113-5f5c9c8044e2
dists_from_centroids = pairwise(Euclidean(ϵ), sample_data, sample_centroids, dims=1)

# ╔═╡ 19ac0ab6-8212-11eb-27ed-132248a13585
size(sample_centroids), size(sample_data)

# ╔═╡ 756d72ec-8208-11eb-378d-2f9b2b288791
md"""
#### Revising clusters

Let's turn to Step 2, where we compute the new centroids given the cluster assignments.

To develop intuition about filtering, let's look at a toy example consisting of 3 data points and 2 clusters.

"""

# ╔═╡ 9a0f5d4a-8208-11eb-11d0-9742f32e4705
begin
  toy_data = [1. 2. 0.; 0. 0. 0.; 2. 2. 0.]
  toy_centroids = [0.5 0.5 0.; 0. -0.5 0.]

  size(toy_data), size(toy_centroids)
end

# ╔═╡ 23ea71ba-822e-11eb-0086-493cd5a4ae95
pairwise(Euclidean(ϵ), toy_data, toy_centroids, dims=1)

# ╔═╡ 4f06ce52-822e-11eb-1c3f-5777aee2de37
pairwise(Euclidean(ϵ), sparse(toy_data), sparse(toy_centroids), dims=1)

# ╔═╡ e58d0b00-8208-11eb-0414-019ec7c74eec
toy_cluster_assign = assign_clusters(toy_data, toy_centroids)

# => 1st and 3rd datapoint assigned to cluster defined by 1st centroid
# => 2nd datapoint assigned to cluster defined by 2nd centroid

# ╔═╡ f00143fa-820d-11eb-2983-43333f967308
function find_cluster(data, cluster_assign, cluster_num)
  """
  Return datapoints assigned to cluster cluster_num
  """
  ary = map(c -> data[c[1], :],
    filter(c -> c[2] == cluster_num,  cluster_assign)
  )
  Matrix(hcat(ary...)')
end

# ╔═╡ 6ceb9cf8-820c-11eb-1652-99c659469ff9
## find datapoint in 1st cluster
find_cluster(toy_data, toy_cluster_assign, 1)

# ╔═╡ 13440200-8250-11eb-0026-c91c8fdff383
## how many datapoints?
size(find_cluster(toy_data, toy_cluster_assign, 1))[1]

# ╔═╡ 86d3d792-8250-11eb-36c0-33630e0c38ba
function count_cluster_assign(cluster_assign)
	"""
	How many data points per cluster
	"""
	ids = map(c -> c[2], cluster_assign)
	[length(ids[ids .== id]) for id ∈ sort(unique(ids))]
end

# ╔═╡ 509e66ca-8252-11eb-28da-dd665ec5575a
count_cluster_assign(sample_cluster_assign)

# ╔═╡ 58c012f2-8252-11eb-3b9a-6b736099ad9d
count_cluster_assign(toy_cluster_assign)

# ╔═╡ c5137efa-820c-11eb-3773-057e25e6c242
## find datapoint in 2nd cluster
find_cluster(toy_data, toy_cluster_assign, 2)

# ╔═╡ 91866190-8213-11eb-00f2-036a1e31e428
## using sparse matrix -> cluster 3
begin
  _s = find_cluster(sample_data, sample_cluster_assign, 3)
  @assert size(_s)[1] ≥ 0

  size(_s), _s
end

# ╔═╡ 33b957ac-820e-11eb-17d6-a35b4a60fb2d
md"""
Given all the data points in a cluster, it only remains to compute the mean.

Use `mean()` from `Statistics` package.
"""

# ╔═╡ 338394f8-820e-11eb-0db4-1f86c79eacbc
mean(find_cluster(toy_data, toy_cluster_assign, 1), dims=1)

# ╔═╡ ab856686-8213-11eb-17e5-730b0a96ad12
## change cluster_num
mean(find_cluster(sample_data, sample_cluster_assign, 3), dims=1)

# ╔═╡ 14fc4e02-8211-11eb-3f9d-d37995fb2522
function revise_centroids(data, k, cluster_assign)
  	n_centroids = []
  	n = size(data)[2]

  	for ix ∈ 1:k
		## 1. Select all data points that belong to cluster ix
        datapts_clu_ix = find_cluster(data, cluster_assign, ix)

    	# size(datapts_clu_ix)[1] == 0 && continue

    	## 2. Compute the mean of the data points.
        centroid = mean(datapts_clu_ix, dims=1)

    	push!(n_centroids, centroid)
  	end
  	reshape(vcat(n_centroids...), k, :)
end

# ╔═╡ 7d692dae-821c-11eb-2aef-97b108c8d8a5
begin
  xv = get_init_centroids(ya_tf_idf, 3; seed=42)
  size(xv), typeof(xv), xv
end

# ╔═╡ c6bb718c-821f-11eb-072c-bf89e4b16e6a
size(toy_data), size(sample_data)

# ╔═╡ e51692fe-8208-11eb-1fc9-1f8f3a9d5db6
begin
  vs1 = revise_centroids(toy_data, 2, toy_cluster_assign)
  size(vs1), typeof(vs1), vs1
end

# ╔═╡ 9330b2f2-8211-11eb-205e-7db725dfdf85
begin
  vs2 = revise_centroids(sample_data, 3, sample_cluster_assign)
  size(vs2), typeof(vs2), vs2
end

# ╔═╡ 92f9c26c-8211-11eb-1f71-818d8fd916fb
md"""
#### Assessing the convergence

How can we tell if the k-means algorithm is converging?

We can look at the cluster assignments and see if they stabilize over time. In fact, we'll be running the algorithm until the cluster assignments stop changing at all.

To be extra safe, and to assess the clustering performance, we'll be looking at an additional criteria: *the sum of all squared distances between data points and centroids*.

This is defined as:

$$J(\mathcal{Z},\mu) = \sum_{j=1}^k \sum_{i:z_i = j} \|\mathbf{x}_i - \mu_j\|^2.$$


The smaller the distances, the more homogeneous the clusters are. In other words, we'd like to have "tight" clusters.
"""

# ╔═╡ 92e15d44-8211-11eb-2977-8d3b271553e1
function compute_heterogeneity(data, k, centroids, cluster_assign)
    heterogeneity = zero(eltype(data))

    for ix ∈ 1:k
    	## Select all data points that belong to cluster ix
        datapts_clu_ix = find_cluster(data, cluster_assign, ix)

    	## check if i-th cluster is non-empty
        if size(datapts_clu_ix)[1] > 0
      		## Compute distances from centroid to data points
      		m_centroid_ix = reshape(centroids[ix, :], 1, :)
        	dists = pairwise(Euclidean(ϵ), datapts_clu_ix, m_centroid_ix, dims=1)
        	heterogeneity += sum(dists .^ 2)
    	end
  	end
    heterogeneity
end

# ╔═╡ 92c7d752-8211-11eb-1546-cd510e3d8659
compute_heterogeneity(sample_data, 2, sample_centroids, sample_cluster_assign)

# ╔═╡ 92accbd8-8211-11eb-0703-0bb21268d6bc
md"""
#### Combining into a single function

Once the two k-means steps have been implemented, as well as our heterogeneity metric we wish to monitor, it is only a matter of putting these functions together to write a k-means algorithm that

  -  Repeatedly performs Steps 1 and 2
  - Tracks convergence metrics
  -  Stops if either no assignment changed or we reach a certain number of iterations.
"""

# ╔═╡ c43df0a0-8216-11eb-3be2-0f69c1faaba2
function kmeans(data, k, init_centroids;
    maxiter=100, record_heterogeneity=nothing, verbose=false)
    """
  This function runs k-means on given data and initial set of centroids.

  - maxiter: maximum number of iterations to run.
    - record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations if nothing, do not store the history.
    - verbose: if true, print how many data points changed their cluster labels in each iteration
  """
    centroids = copy(init_centroids)
    prev_cluster_assign, cluster_assign = nothing, nothing

    for ix ∈ 1:maxiter
        verbose && print("\niter: $(ix) ")

        ## 1. Make cluster assignments using nearest centroids
        cluster_assign = assign_clusters(data, centroids)

    	## 2. Compute a new centroid for each of the k clusters,
    	##    averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assign)

    	## 3. Check for convergence: if none of the assignments changed, stop
        if !isnothing(prev_cluster_assign) &&
          all(t -> t[1] == t[2], zip(prev_cluster_assign, cluster_assign))
      		break
    	end

    	## 4. Print number of new assignments
        if !isnothing(prev_cluster_assign)
            num_changed = sum(prev_cluster_assign .≠ cluster_assign)
            verbose && 
			  @printf("\t%5d elements changed their cluster assignment.\n", num_changed)
    	end

        if !isnothing(record_heterogeneity)
      		## 5. Record heterogeneity convergence metric
            score = compute_heterogeneity(data, k, centroids, cluster_assign)
            push!(record_heterogeneity, score)
    	end

        prev_cluster_assign = copy(cluster_assign) ## Copy (and not a ref.)
  	end
    (centroids, cluster_assign)
end

# ╔═╡ c422dff2-8216-11eb-2f63-235d58c77e35
md"""
##### Plotting the convergence
"""

# ╔═╡ c406c684-8216-11eb-3aba-65c9a55b7441
function plot_heterogeneity(heterogeneity, k)
    # plt.figure(figsize=(7,4))
 	plot(heterogeneity,
    	linewidth=4,
    	xlabel="# Iterations",
    	ylabel="Heterogeneity",
    	title="Heterogeneity of clustering over time, K=$(k)"
 	)
end

# ╔═╡ c3ebe820-8216-11eb-06e1-538f0cda84b7
begin
  k, heterogeneity = 3, []
  init_centroids = get_init_centroids(norm_ya_tf_idf, k; seed=42)
  centroids, cluster_assign = nothing, nothing

  with_terminal() do
    global centroids, cluster_assign = kmeans(norm_ya_tf_idf, k, init_centroids;
        maxiter=400, record_heterogeneity=heterogeneity, verbose=true);
  end
end

# ╔═╡ 31a82b96-8234-11eb-2795-bbadf977d8a3
centroids

# ╔═╡ bf1fd5b6-8234-11eb-0fbc-91da7f6dbe28
heterogeneity

# ╔═╡ 3633fa2a-8234-11eb-315d-d93094e52cab
size(find_cluster(norm_ya_tf_idf, cluster_assign, 1))

# ╔═╡ 67ecd7b2-8234-11eb-03f5-41fd8adda1fd
size(find_cluster(norm_ya_tf_idf, cluster_assign, 2))

# ╔═╡ 81b59776-8234-11eb-03d8-9736ea84e866
size(find_cluster(norm_ya_tf_idf, cluster_assign, 3))

# ╔═╡ a6d93074-8234-11eb-3cba-f1d9d8dbdfe3
plot_heterogeneity(heterogeneity, k)

# ╔═╡ 30eb64d4-824a-11eb-06c6-3393ca6a5780
md"""
#### Beware of local maxima

One weakness of k-means is that it tends to get stuck in a local minimum. To see this, let us run k-means multiple times, with different initial centroids created using different random seeds.

Note: Again, in practice, you should set different seeds for every run. We give you a list of seeds for this assignment so that everyone gets the same answer.

This may take several minutes to run.
"""

# ╔═╡ 2c29b028-8253-11eb-2781-f3de2eb49ba3
now()

# ╔═╡ 4c3ed812-824a-11eb-356a-6594a7b2a65e
function run_kmeans_with_multi_seeds(tfidf; k=10, init=:random)
 	heterogeneity = Dict()
  	cluster_assign_h = Dict()
  	# start = now()
  	m_ix_largest_clu, m_larg_clu_sz = -1, -1
  
  	for seed ∈ [0, 2000, 4000, 6000, 8000, 10000, 12000]
    	init_centroids = init == :random ? get_init_centroids(tfidf, k; seed) :
			smart_initialize(tfidf, k, seed)
		
    	centroids, cluster_assign = kmeans(tfidf, k, init_centroids;
			maxiter=400, record_heterogeneity=nothing, verbose=false)
    
		## To save time, compute heterogeneity only once in the end
    	heterogeneity[seed] = compute_heterogeneity(tfidf, k, centroids, 	 
			cluster_assign)
    	cluster_assign_h[seed] = count_cluster_assign(cluster_assign)
		
    	ix_larg_clu = argmax(cluster_assign_h[seed])
    	larg_clu_sz = cluster_assign_h[seed][ix_larg_clu]
		
    	@printf("%s seed=%6d, heterogeneity=%.5f ", string(now()), 
			seed, heterogeneity[seed])
        println("cluster_distribution=$(cluster_assign_h[seed]) / largest cluster: $(ix_larg_clu) size: $(larg_clu_sz)")
		
    	if larg_clu_sz > m_larg_clu_sz
       		m_ix_largest_clu, m_larg_clu_sz = ix_larg_clu, larg_clu_sz
		end
	end
  	@printf("took: %s\n", string(now()))
  	(cluster_assign_h, heterogeneity)
end

# ╔═╡ 175aa3f2-824c-11eb-01aa-f789353345f4
## with random init
begin
	cluster_assign_wri, heterogeneity_wri = nothing, nothing
	
	with_terminal() do
		global cluster_assign_wri, heterogeneity_wri = 
			run_kmeans_with_multi_seeds(norm_ya_tf_idf)
	end
end

# ╔═╡ Cell order:
# ╟─f701bf4a-809e-11eb-2f1f-bb7dad75bfcc
# ╠═2a641766-809f-11eb-274a-8fdb234c0d08
# ╠═692a39c8-80a4-11eb-2077-e9f391787416
# ╠═5b3b148e-809f-11eb-3631-cf4682cbd959
# ╠═97d6cb18-809f-11eb-075a-7b3c5354ce4c
# ╠═97d2bd70-8239-11eb-2a8c-efea2584e6d6
# ╠═54ce09fa-8246-11eb-35fc-a9ba218d8c76
# ╟─6ada72c6-80a0-11eb-2346-c3554265a4de
# ╠═9ee529a8-80a0-11eb-23cc-b78ead6503e4
# ╠═16431f28-80a4-11eb-12a2-f3104b93f673
# ╟─e98e9a5e-80a3-11eb-1d38-db4ff0f9575b
# ╠═fbebf142-80a3-11eb-0913-4b43d543a681
# ╠═637c61ac-80a4-11eb-21be-f1661d34b568
# ╟─a30ee624-81e8-11eb-0c75-a5715165ff36
# ╠═4543b590-80a5-11eb-3215-b9bb37e1e766
# ╠═fd36c16e-80a6-11eb-3a80-5d7b8cb48b38
# ╠═4e94e7a2-80a7-11eb-0fb3-1955f645d0ca
# ╠═107cca2e-80ad-11eb-19b8-bf015a43b184
# ╠═69c34f00-80ac-11eb-0b03-0362d6f5e84e
# ╠═b0d25bf0-80ae-11eb-00de-d57e0d22a000
# ╟─07757e58-8247-11eb-299a-97cbb8ce1460
# ╠═99a6e9c4-8238-11eb-3acc-63a515a49b64
# ╠═3f2f94e4-80af-11eb-32d2-a5e101797f53
# ╠═fcfefb98-81e7-11eb-014a-b72b0477acab
# ╠═6c7e5bb4-823b-11eb-3833-254b4f5dafa7
# ╟─aaea9810-81e9-11eb-0452-83d979dca4c3
# ╠═aad0780c-81e9-11eb-2344-63996006b0cb
# ╟─446d44fa-81eb-11eb-0daa-83db97d8eec9
# ╟─f1494764-81eb-11eb-02e8-e9a44cbbd905
# ╠═444fc56a-81eb-11eb-27d2-1b249cae06d9
# ╠═9716ee8a-81fb-11eb-3279-c19f1a5736a2
# ╠═95d6fcdc-81ff-11eb-0d5b-5b161dba559f
# ╠═68962f52-8215-11eb-1cf2-1763b3ea2fcf
# ╠═aab3edea-81e9-11eb-2104-f18b12260e09
# ╟─aa994aa0-81e9-11eb-1fca-f1b45c0220f0
# ╠═8f71f718-8200-11eb-2dc1-4bc972d149f2
# ╠═20c45328-8237-11eb-1cf3-a9e5a4ee496f
# ╠═f812155c-8232-11eb-0113-5f5c9c8044e2
# ╠═19ac0ab6-8212-11eb-27ed-132248a13585
# ╟─756d72ec-8208-11eb-378d-2f9b2b288791
# ╠═9a0f5d4a-8208-11eb-11d0-9742f32e4705
# ╠═23ea71ba-822e-11eb-0086-493cd5a4ae95
# ╠═4f06ce52-822e-11eb-1c3f-5777aee2de37
# ╠═e58d0b00-8208-11eb-0414-019ec7c74eec
# ╠═f00143fa-820d-11eb-2983-43333f967308
# ╠═6ceb9cf8-820c-11eb-1652-99c659469ff9
# ╠═13440200-8250-11eb-0026-c91c8fdff383
# ╠═86d3d792-8250-11eb-36c0-33630e0c38ba
# ╠═509e66ca-8252-11eb-28da-dd665ec5575a
# ╠═58c012f2-8252-11eb-3b9a-6b736099ad9d
# ╠═c5137efa-820c-11eb-3773-057e25e6c242
# ╠═91866190-8213-11eb-00f2-036a1e31e428
# ╟─33b957ac-820e-11eb-17d6-a35b4a60fb2d
# ╠═338394f8-820e-11eb-0db4-1f86c79eacbc
# ╠═ab856686-8213-11eb-17e5-730b0a96ad12
# ╠═14fc4e02-8211-11eb-3f9d-d37995fb2522
# ╠═7d692dae-821c-11eb-2aef-97b108c8d8a5
# ╠═c6bb718c-821f-11eb-072c-bf89e4b16e6a
# ╠═e51692fe-8208-11eb-1fc9-1f8f3a9d5db6
# ╠═9330b2f2-8211-11eb-205e-7db725dfdf85
# ╟─92f9c26c-8211-11eb-1f71-818d8fd916fb
# ╠═92e15d44-8211-11eb-2977-8d3b271553e1
# ╠═92c7d752-8211-11eb-1546-cd510e3d8659
# ╟─92accbd8-8211-11eb-0703-0bb21268d6bc
# ╠═c43df0a0-8216-11eb-3be2-0f69c1faaba2
# ╟─c422dff2-8216-11eb-2f63-235d58c77e35
# ╠═c406c684-8216-11eb-3aba-65c9a55b7441
# ╠═c3ebe820-8216-11eb-06e1-538f0cda84b7
# ╠═31a82b96-8234-11eb-2795-bbadf977d8a3
# ╠═bf1fd5b6-8234-11eb-0fbc-91da7f6dbe28
# ╠═3633fa2a-8234-11eb-315d-d93094e52cab
# ╠═67ecd7b2-8234-11eb-03f5-41fd8adda1fd
# ╠═81b59776-8234-11eb-03d8-9736ea84e866
# ╠═a6d93074-8234-11eb-3cba-f1d9d8dbdfe3
# ╟─30eb64d4-824a-11eb-06c6-3393ca6a5780
# ╠═2c29b028-8253-11eb-2781-f3de2eb49ba3
# ╠═4c3ed812-824a-11eb-356a-6594a7b2a65e
# ╠═175aa3f2-824c-11eb-01aa-f789353345f4
