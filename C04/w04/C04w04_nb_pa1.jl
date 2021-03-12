### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 1f62d332-8303-11eb-31e8-4b34a42c4b9f
begin
 	using Pkg
  	Pkg.activate("MLJ_env", shared=true)

	# using CSV
  	# using DataFrames
 	using PlutoUI
  	using Test
  	using Printf
  	using Plots
  	using LinearAlgebra
  	# using SparseArrays
  	# using TextAnalysis
  	using Random
  	# using Distances
  	using Statistics
	using Distributions
  	# using Dates
end

# ╔═╡ d67867cc-8302-11eb-3d40-d5a50515cdb9
md"""
## C04w04: Fitting Gaussian Mixture Models with EM

In this assignment we will
  - implement the EM algorithm for a Gaussian mixture model
  - apply your implementation to cluster images
  - explore clustering results and interpret the output of the EM algorithm  
"""

# ╔═╡ 53e17eba-8303-11eb-2471-eb8521bf88d2
md"""
### Implementing the EM algorithm for Gaussian mixture models

In this section, we will implement the EM algorithm taking the following steps:
  - Provide a log likelihood function for this model.
  - Implement the EM algorithm.
  - Create some synthetic data.
  - Visualize the progress of the parameters during the course of running EM.
  - Visualize the convergence of the model.
"""

# ╔═╡ 68c8f63a-8303-11eb-0ae3-c3f39e9acf8d
md"""
#### Log likelihood

We provide a function to calculate log likelihood for mixture of Gaussians. The log likelihood quantifies the probability of observing a given set of data under a particular setting of the parameters in our model. 

We will use this to assess the convergence of our EM algorithm; specifically, we will keep looping through EM update steps until the log likehood ceases to increase at a certain rate.
"""

# ╔═╡ 84b9158c-8303-11eb-2ce4-c109ce3ff51a
function log_sum_exp(z)
    """
    Compute log(Σᵢ exp(zᵢ)) for some array z.
    """
    max_z = maximum(z)
    max_z + log(sum(exp.(z .- max_z)))
end

# ╔═╡ c01ba57e-8303-11eb-34e7-dfb911bac2f2
function log_likelihood(data, weights, μ, σ)
    """
    Compute the log likelihood of the data for a Gaussian mixture model 
	with the given parameters.
    """
    nclusters, ndim = length(means), size(data)[1]
	DT = eltype(data)
    llh = zero(DT)
	
    for d ∈ data
        z = zeros(DT, nclusters)
        for k ∈ 1:nclusters   
            ## Compute: (x - μ)ᵀ ×  Σ⁻¹ × (x - μ)
            δ = d .- μ[k]
            exp_term = dot(delta', dot(inv(σ[k]), δ))
            
            ## Compute log likelihood contribution for this data point 
			## and this cluster
            z[k] += log(weights[k])
            z[k] -= 1/2. * (n_dim * log(2. * π) + log(det(σ[k])) + expo_term)
		end
        ## Increment llh contribution of this data point across all clusters
        llh += log_sum_exp(z)
	end
    llh
end

# ╔═╡ 580675c0-830a-11eb-16e4-b963a391b138
md"""
#### E-step: assign cluster responsibilities, given current parameters

The first step in the EM algorithm is to compute cluster responsibilities.  

Let $r_{ik}$ denote the responsibility of cluster $k$ for data point $i$.
Note that cluster responsibilities are fractional parts. *Cluster responsibilities for a single data point $i$ should sum to 1*.

$$r_{i1} + r_{i2} + \ldots + r_{iK} = 1$$

To figure how much a cluster is responsible for a given data point, we compute the likelihood of the data point under the  particular cluster assignment, multiplied by the weight of the cluster. <br />
For data point $i$ and cluster $k$, this quantity is

$$r_{ik} \propto \pi_k N(x_i | \mu_k, \Sigma_k)$$

where $N(x_i | \mu_k, \Sigma_k)$ is the Gaussian distribution for cluster $k$ (with mean $\mu_k$ and covariance $\Sigma_k$).

We used $\propto$ because the quantity $N(x_i | \mu_k, \Sigma_k)$ is not yet the responsibility we want. <br /> To ensure that all responsibilities over each data point add up to 1, we add the normalization constant in the denominator:

$$r_{ik} = \frac{\pi_k N(x_i | \mu_k, \Sigma_k)}{\sum_{k=1}^{K} \pi_k N(x_i | \mu_k, \Sigma_k)}$$

Complete the following function that computes $r_{ik}$ for all data points $i$ and clusters $k$.
"""

# ╔═╡ 4d69bcec-830c-11eb-0c80-314c8cded6db
function compute_responsibilities(data, weights, μ, σ)
    """
    E-step fo EM Algorithm: 
    - compute responsibilities, given the current parameters
    """
	@assert length(data) > 0
	
    ndata, nclusters = length(data), length(μ)
    resp = zeros(eltype(data[1]), ndata, nclusters)
    
    ## Update resp. matrix so that resp[i, k] is the responsibility of 
	## cluster k for data point i.
	
    for i ∈ 1:ndata
        for k ∈ 1:nclusters
			d = MvNormal(μ[k], σ[k])
            resp[i, k] = weights[k] * pdf(d, data[i])
		end
	end
    ## Add up responsibilities over each data point and normalize
    row_sums = reshape(sum(resp, dims=2), :, 1) 
    resp ./= row_sums
end

# ╔═╡ 31835154-830d-11eb-09fd-196046609868
begin
	const ϵ = 1e-8
	resp₀ = compute_responsibilities([[1., 2.], [-1., -2.]], 
		[0.3, 0.7],                         # weights
		[[0., 0.], [1., 1.]],               # μ
        [[1.5 0.; 0. 2.5], [1. 1.; 1. 2.]]  # σ
	)

	@test size(resp₀) == (2, 2)
	@test all(t -> abs(t[1] - t[2]) ≤ ϵ, 
		zip(resp₀, [0.10512733 0.89487267; 0.46468164 0.53531836]))
end

# ╔═╡ 430e58a8-8310-11eb-2a53-ed2608f43122
md"""
#### M-step: Update parameters, given current cluster responsibilities

Once the cluster responsibilities are computed, we update the parameters (*weights*, *means*, and *covariances*) associated with the clusters.

**Computing soft counts**. Before updating the parameters, we first compute what is known as "soft counts".

The soft count of a cluster is the sum of all cluster responsibilities for that cluster:

$$N^{\text{soft}}_k = r_{1k} + r_{2k} + \ldots + r_{Nk} = \sum_{i=1}^{N} r_{ik}$$

where we loop over data points. Note that, unlike k-means, we must loop over every single data point in the dataset. This is because all clusters are represented in all data points, to a varying degree.

"""

# ╔═╡ 42ef055c-8310-11eb-1ae5-d5ae8bf22c1f
"""
  Compute the total responsibility assigned to each cluster, which will be useful when 
  implementing M-steps below. In the lectures this is called N^soft
"""
compute_soft_counts(resp) = sum(resp; dims=1)

# ╔═╡ 93988ed8-8310-11eb-15f4-554b537eacd6
md"""
**Updating weights.** The cluster weights show us how much each cluster is represented over all data points.

The weight of cluster $k$ is given by the ratio of the soft count 
$N^{\text{soft}}_{k}$ to the total number of data points $N$:

$$\hat{\pi}_k = \frac{N^{\text{soft}}_{k}}{N}$$

Notice that $N$ is equal to the sum over the soft counts $N^{\text{soft}}_{k}$ of all clusters.

Complete the following function:
"""

# ╔═╡ 937da226-8310-11eb-1966-6bcc8ff3cb47
function compute_weights(counts)
    nclusters = length(counts)
    weights = zeros(eltype(counts), nclusters)
    n = sum(counts)
	for k ∈ 1:nclusters
        ## Update the weight for cluster k using the M-step update rule for 
		## the cluster weight: π̂ₖ
        weights[k] = counts[k] / n
	end
    weights
end

# ╔═╡ 93653664-8310-11eb-23f2-d5236b8bc0ed
begin
	resp₁ = compute_responsibilities([[1., 2.], [-1.,-2.], [0., 0.]], 
		[0.3, 0.7],                         # weights     
		[[0., 0.], [1., 1.]],               # μ
		[[1.5  0.; 0. 2.5], [1. 1.; 1. 2.]] # σ
	)
	counts₁ = compute_soft_counts(resp₁)
	weights₁ = compute_weights(counts₁)
	
	@test all(t -> abs(t[1] - t[2]) ≤ ϵ,
		zip(weights₁, [0.27904865942515705 0.720951340574843]))
end

# ╔═╡ 93490854-8310-11eb-3de5-5b24918d8840
with_terminal() do
	println(" - counts: $(counts₁)")	
	println(" - weigths: $(weights₁)")
	println("Checkpoint passed!")
end

# ╔═╡ 42d1c078-8310-11eb-2262-bb5ae92a2b27
md"""
**Updating means**. The mean of each cluster is set to the [weighted average](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) of all data points, weighted by the cluster responsibilities:

$$\hat{\mu}_k = \frac{1}{N_k^{\text{soft}}} \sum_{i=1}^N r_{ik}x_i$$

Complete the following function:
"""

# ╔═╡ 1b27764e-8315-11eb-3854-f9e5b591744a
function compute_means(data, resp, counts)
    nclusters, ndata = length(counts), length(data)
    means = [zeros(eltype(data[1]), length(data[1]))  for _ ∈ 1:nclusters]
    
    for k in 1:nclusters
        ## Update means for cluster k using the M-step update rule 
		## for the mean variables.
        ## This will assign the variable means[k] to be our estimate for μ̂ₖ.
        weighted_sum = sum(
          [resp[i, k] * data[i] for i ∈ 1:ndata]
        )
        means[k] = weighted_sum / counts[k]
	end
    means
end

# ╔═╡ 34b1bb5e-8317-11eb-0260-5f1d3132a7ba
begin
	data_tmp = [[1., 2.], [-1.,-2.]]
	resp₂ = compute_responsibilities(data_tmp, 
		[0.3, 0.7],                         # weights     
		[[0., 0.], [1., 1.]],               # μ
        [[1.5  0.; 0. 2.5], [1. 1.; 1. 2.]] # σ
	)

	counts₂ = compute_soft_counts(resp₂)
	means₂ = compute_means(data_tmp, resp₂, counts₂)

	 @test all(((x, y)=t) -> abs(x[1] - y[1]) ≤ ϵ && abs(x[2] - y[2]) ≤ ϵ,
		zip(means₂, [[-0.6310085, -1.262017], [0.25140299, 0.50280599]]))
	
end

# ╔═╡ 1ad0cbe4-8315-11eb-341c-dbce64acb374
with_terminal() do
	println(" - counts: $(counts₂)")
	println(" - means: $(means₂)")
	println("Checkpoint passed!")
end

# ╔═╡ 1a99af8a-8315-11eb-0f74-eb97741a40ca
md"""

**Updating covariances**.  The covariance of each cluster is set to the weighted average of all [outer products](https://people.duke.edu/~ccc14/sta-663/LinearAlgebraReview.html), weighted by the cluster responsibilities:

$$\hat{\Sigma}_k = \frac{1}{N^{\text{soft}}_k}\sum_{i=1}^N r_{ik} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

The "outer product" in this context refers to the matrix product

$$(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T.$$
Letting $(x_i - \hat{\mu}_k)$ to be $d \times 1$ column vector, this product is a $d \times d$ matrix. Taking the weighted average of all outer products gives us the covariance matrix, which is also $d \times d$.

Reminder:  
The inner product (of two column vectors $1 \times n$) denoted as $<v, w> = v^{t}w$ is a scalar, whereas the outer product of the same two column vectors denoted as $v \otimes w = vw^{t}$ is a matrix.

Complete the following function:

"""

# ╔═╡ 986e362c-8317-11eb-0eef-cd1de11cd4b2
# TODO

# ╔═╡ Cell order:
# ╟─d67867cc-8302-11eb-3d40-d5a50515cdb9
# ╠═1f62d332-8303-11eb-31e8-4b34a42c4b9f
# ╟─53e17eba-8303-11eb-2471-eb8521bf88d2
# ╟─68c8f63a-8303-11eb-0ae3-c3f39e9acf8d
# ╠═84b9158c-8303-11eb-2ce4-c109ce3ff51a
# ╠═c01ba57e-8303-11eb-34e7-dfb911bac2f2
# ╟─580675c0-830a-11eb-16e4-b963a391b138
# ╠═4d69bcec-830c-11eb-0c80-314c8cded6db
# ╠═31835154-830d-11eb-09fd-196046609868
# ╟─430e58a8-8310-11eb-2a53-ed2608f43122
# ╠═42ef055c-8310-11eb-1ae5-d5ae8bf22c1f
# ╟─93988ed8-8310-11eb-15f4-554b537eacd6
# ╠═937da226-8310-11eb-1966-6bcc8ff3cb47
# ╠═93653664-8310-11eb-23f2-d5236b8bc0ed
# ╠═93490854-8310-11eb-3de5-5b24918d8840
# ╟─42d1c078-8310-11eb-2262-bb5ae92a2b27
# ╠═1b27764e-8315-11eb-3854-f9e5b591744a
# ╠═34b1bb5e-8317-11eb-0260-5f1d3132a7ba
# ╠═1ad0cbe4-8315-11eb-341c-dbce64acb374
# ╟─1a99af8a-8315-11eb-0f74-eb97741a40ca
# ╠═986e362c-8317-11eb-0eef-cd1de11cd4b2
