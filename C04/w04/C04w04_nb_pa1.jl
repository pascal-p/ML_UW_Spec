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
    nclusters, ndim = length(μ), size(data)[1]
	DT = eltype(data[1])
    llh = zero(DT)
	
    for d ∈ data
        z = zeros(DT, nclusters)
        for k ∈ 1:nclusters   
            ## Compute: (x - μ)ᵀ ×  Σ⁻¹ × (x - μ)
            δ = reshape(d .- μ[k], :, 1) # make sure 2-dim!	
            exp_term = dot(δ', inv(σ[k]) * δ)
            
            ## Compute log likelihood contribution for this data point 
			## and this cluster
            z[k] += log(weights[k])
            z[k] -= 1/2. * (ndim * log(2. * π) + log(det(σ[k])) + exp_term)
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
    means = [zeros(eltype(data[1]), length(data[1])) for _ ∈ 1:nclusters]
    
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

	exp_means₂ = [[-0.6310085, -1.262017], [0.25140299, 0.50280599]]
	
	@test all(t -> abs(t[1] - t[2]) ≤ ϵ, zip(means₂[1], exp_means₂[1]))
	@test all(t -> abs(t[1] - t[2]) ≤ ϵ, zip(means₂[2], exp_means₂[2]))
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
function compute_covariances(data, resp, counts, μ)
    nclusters, ndim, ndata = length(counts), size(data[1])[1], length(data)
    covs = [zeros(eltype(data[1]), ndim, ndim) for _ ∈ 1:nclusters]
    
    for k ∈ 1:nclusters
        ## Update covariances for cluster k using the M-step update rule 
		## for covariance variables.
        ## This will assign the variable covariances[k] to be the estimate for Σ̂ₖ.
        w_sum = zeros(eltype(data[1]), ndim, ndim)
        for i ∈ 1:ndata
            w_sum += resp[i, k] * ((data[i] - μ[k]) * (data[i] - μ[k])')
		end
		covs[k] = (1. / counts[k]) * w_sum
	end
    covs
end

# ╔═╡ 57c76562-8373-11eb-0e43-256ec189f399
begin
	## deps on prev test cell for resp₂, counts₂, means₂
	covs₂ = compute_covariances(data_tmp, resp₂, counts₂, means₂)
	
	exp_covs₂ = [[0.93679654, 1.87359307], [1.87359307, 3.74718614]]
	
	@test all(t -> abs(t[1] - t[1]) ≤ ϵ, zip(covs₂[1], exp_covs₂[1]))
	@test all(t -> abs(t[1] - t[1]) ≤ ϵ, zip(covs₂[2], exp_covs₂[2]))
end

# ╔═╡ a12bc08e-8371-11eb-29f0-e14f2c17c1d1
with_terminal() do
	println(" - covariances: $(covs₂)")
	println("Checkpoint passed!")
end

# ╔═╡ c59d7e92-8373-11eb-09d1-55814d7e9f9e
md"""
#### The EM algorithm

Now let us write a function that takes initial parameter estimates and runs EM.
"""

# ╔═╡ c5808488-8373-11eb-3d9a-51c955524390
function em(data, init_μ, init_σ, init_weights;
		maxiter=1000, ϵ=1e-4)
    ## Make copies of initial parameters, which we will update during each iteration
    μ, σ = copy(init_μ), copy(init_σ)
    weights = copy(init_weights)
    
    ## Infer dimensions of dataset and the number of clusters
    ndata, ndim, nclusters = length(data), size(data[1])[1], length(μ)
    
    ## Initialize some useful variables
    resp = zeros(eltype(data[1]), ndata, nclusters)
    llh = log_likelihood(data, weights, μ, σ)
    llh_trace = [llh]
    
    for it ∈ 1:maxiter
        it % 5 == 0 && println("Iteration $(it) / $(maxiter)")
        
        ## E-step: compute responsibilities
        resp = compute_responsibilities(data, weights, μ, σ)

        ## M-step
        ## Calc. the total resp. assigned to each cluster, useful when 
        ## implementing M-steps below. In the lectures this is called N^{soft}
        counts = compute_soft_counts(resp)
        ## update weight of cluster k using M-step update rule for cluster weight π̂ₖ
        weights = compute_weights(counts)
        ## update means of cluster k using the M-step update rule for the mean vars.
        ## assign the variable μ[k] to be our estimate for μ̂ₖ.
        μ = compute_means(data, resp, counts)
        ## update covs ofcluster k using the M-step update rule for cov vars.
        ## assign the variable σ[k] to be the estimate for Σ̂ₖ.
        σ = compute_covariances(data, resp, counts, μ)
        
        ## Compute the loglikelihood at this iteration
        llh_latest = log_likelihood(data, weights, μ, σ)
        push!(llh_trace, llh_latest)
        
        ## Check for convergence in log-likelihood and store
        (llh_latest - llh) < ϵ && llh_latest > Base.Inf && break
        llh = llh_latest
	end
    Dict{Symbol, Any}(
		:weights => weights, 
		:means => μ, 
		:covs => σ, 
		:loglik => llh_trace, 
		:resp => resp
	)
end

# ╔═╡ c56791bc-8373-11eb-00e3-230f82206892
md"""
#### Testing the implementation on the simulated data

To help us develop and test our implementation, we will generate some observations from a mixture of Gaussians and then run our EM algorithm to discover the mixture components. 
We'll begin with a function to generate the data, and a quick plot to visualize its output for a 2-dimensional mixture of three Gaussians.

Now we will create a function to generate data from a mixture of Gaussians model. 
"""

# ╔═╡ a790272e-8375-11eb-2a40-df02f3d1ca24
function generate_mog_data(ndata, μ, σ, weights; seed=42)
    """
    Creates a list of data points
    """
    nclusters = length(weights)
    data = []
	
	Random.seed!(seed)	
	rng = MersenneTwister(seed)
	
    for i ∈ 1:ndata
        ## Use np.random.choice and weights to pick a: 1 ≤ cluster_id < nclusters
        k = rand(rng, 1:nclusters)
        
		## Use np.random.multivariate_normal to create data from this cluster
		x = rand(MvNormal(μ[k], σ[k]))  ## ????
		
		# np.random.multivariate_normal(means[k], covariances[k])
        push!(data, x)
	end
    data
end

# ╔═╡ a740cde6-8375-11eb-0f2a-efc1f23f8f38
begin
	init_μ = [
    	[5., 0.], # mean of cluster 1
    	[1., 1.], # mean of cluster 2
    	[0., 5.]  # mean of cluster 3
	]

	init_σ =[
		[.5 0.; 0 .5],       # covariance of cluster 1
    	[.92 .38; .38 .91],  # covariance of cluster 2
    	[.5 0.; 0 .5]]       # covariance of cluster 3

	init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster
	
	data = generate_mog_data(100, init_μ, init_σ, init_weights)
end

# ╔═╡ a72ae382-8375-11eb-15fc-c750a071dccf
scatter(map(t -> t[1], data), map(t -> t[2], data), color=[:lightblue], lw=3)

# ╔═╡ a7135b54-8375-11eb-2d63-392c4fa94aea
md"""
Now we will fit a mixture of Gaussians to this data using our implementation of the EM algorithm. As with k-means, it is important to ask how we obtain an initial configuration of mixing weights and component parameters. 

In this simple case, we'll take three random points to be the initial cluster means, use the empirical covariance of the data to be the initial covariance in each cluster (a clear overestimate), and set the initial mixing weights to be uniform across clusters.
"""

# ╔═╡ a6fc6796-8375-11eb-02af-01cee06a06dd
begin
	seed₂ = 2010
	Random.seed!(seed₂)	
	rng = MersenneTwister(seed₂)
	
	## Initialization of parameters
	const K = 3
	
	## FIXME: review
	chosen = randperm(rng, length(data))[1:K] 
	#        np.random.choice(length(data), K, replace=False)
	
	@assert length(chosen) == K
	
	init_μ₁ = [data[x] for x in chosen]
	init_σ₁ = [cov(data) for _ ∈ 1:K] # cov(data, rowvar=0) => col. is an observ. 
	init_w = [1.0 / K for _ ∈ 1:K]

	## Run EM 
	results = em(data, init_μ₁, init_σ₁, init_w)
end

# ╔═╡ a6b33828-8375-11eb-027a-130ef7a14128
md"""
**Note**. Like k-means, EM is prone to converging to a local optimum. In practice, you may want to run EM multiple times with different random initialization. We have omitted multiple restarts to keep the assignment reasonably short.


Our algorithm returns a dictionary with five elements: 
  - 'loglik': a record of the log likelihood at each iteration
  - 'resp': the final responsibility matrix
  - 'means': a list of K means
  - 'covs': a list of K covariance matrices
  - 'weights': the weights corresponding to each model component


*For the following quiz questions, please round your answer to the nearest thousandth (3 decimals)*

**Quiz Question**: What is the weight that EM assigns to the first component after running the above codeblock?
"""

# ╔═╡ a69ff718-8375-11eb-3965-89ecd2088d97
round(results[:weights][1]; digits=3)

# ╔═╡ a687f546-8375-11eb-3435-1fa822f71ff1
md"""
**Quiz Question**: Using the same set of results, obtain the mean that EM assigns the second component. What is the mean in the first dimension?
"""

# ╔═╡ a66ede62-8375-11eb-3746-758cb7efc8a8
## Answer:   the second component / first dim
round(results[:means][2][1], digits=3)

# ╔═╡ a657ed74-8375-11eb-1d97-1bd417fa42ee
md"""
**Quiz Question**: Using the same set of results, obtain the covariance that EM assigns the third component. What is the variance in the first dimension?
"""

# ╔═╡ a63e00e4-8375-11eb-10db-897f94a0dc32
# Answer:   covs third component / first dim
results[:covs][3, 1]

# ╔═╡ c54c0456-8373-11eb-1cb6-11bf225bc6b0
md"""
### Plot progress of parameters

One useful feature of testing our implementation on low-dimensional simulated data is that we can easily visualize the results. 

We will use the following `plot_contours` function to visualize the Gaussian components over the data at three different points in the algorithm's execution:

  1. At initialization (using initial_mu, initial_cov, and initial_weights)
  2. After running the algorithm to completion 
  3. After just 12 iterations (using parameters estimates returned when setting `maxiter=12`)
"""

# ╔═╡ 9ba36afc-83d7-11eb-32d4-4d4d0df107d4
function bivariate_normal(X, Y;
		σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0)
    Xμ = X .- μ_x
    Yμ = Y .- μ_y
	
    ρ = σ_xy / (σ_x * σ_y)
    z = Xμ^2 / σ_x^2 + Yμ^2 / σ_y^2 - 2 * ρ * Xμ * Yμ / (σ_x * σ_y)
    println("=> ", size(Xμ), " / ", size(Yμ), " /  ", size(z))
	
	den = 2. * π * σ_x * σ_y * √(1 - ρ^2)
	r = exp(-z / (2. * (1. - ρ^2))) / den
	println("==",  size(r))
	r
end

# ╔═╡ cd5eb61e-83d9-11eb-1df3-9389091527e8
function meshgrid(x, y)
	@assert length(x) == length(y)
	n = length(x)
	
	vx, vy = copy(x), copy(y)
	for _ ∈ 1:n-1
		vx = hcat(vx, x)
		vy = hcat(vy, y)
	end
	vx, vy = reshape(vx, :, n)', reshape(vy, :, n)'
	Matrix(vx), Matrix(vy)
end

# ╔═╡ 207ec42e-83df-11eb-129f-b5e34bda1328
md"""
FIXME plot_contour...
"""

# ╔═╡ 5a708ed0-83d8-11eb-147c-abf3f80f8c92
function plot_contours(data, means, covs, title)
    δ = 0.025
    k = length(means)
    x = collect(-2.0:δ:7.0) # was 7.0, 6.975
    y = collect(-2.0:δ:7.0)
    X, Y = meshgrid(x, y)
    col = [:green, :red, :indigo]
	Z = []
	println("OK...")
	
	for i ∈ 1:k
		μ, σ = means[i], covs[i]
		σ_x, σ_y = √(σ[1, 1]), √(σ[2, 2])	
		σ_xy = σ[1, 2] / (σ_x * σ_y)
		push!(Z, bivariate_normal(X, Y; σ_x, σ_y, μ_x=μ[1], μ_y=μ[2], σ_xy))
	end
	println("OK1...size(X): ", size(X), " size(X): size(Y): ", size(Y), " size(Z): ",  size(Z[1]))
	
	scatter([_x[1] for _x ∈ data], [_y[2] for _y ∈ data]) # data
	
	contour!(X, Y, Z[1], colors=col[1])
	contour!(X, Y, Z[2], colors=col[2])
	contour!(X, Y, Z[3], colors=col[3])
	
	# title
end

# ╔═╡ 9b4a0a5e-83d7-11eb-3f02-a5ece0576a9b
plot_contours(data, init_μ₁, init_σ₁, "Initial clusters")

# ╔═╡ 9aee7716-83d7-11eb-3f3c-d534adaf4bd6


# ╔═╡ baa2d2f0-8373-11eb-27fd-9fc4b6b46360


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
# ╠═57c76562-8373-11eb-0e43-256ec189f399
# ╠═a12bc08e-8371-11eb-29f0-e14f2c17c1d1
# ╟─c59d7e92-8373-11eb-09d1-55814d7e9f9e
# ╠═c5808488-8373-11eb-3d9a-51c955524390
# ╟─c56791bc-8373-11eb-00e3-230f82206892
# ╠═a790272e-8375-11eb-2a40-df02f3d1ca24
# ╠═a740cde6-8375-11eb-0f2a-efc1f23f8f38
# ╠═a72ae382-8375-11eb-15fc-c750a071dccf
# ╟─a7135b54-8375-11eb-2d63-392c4fa94aea
# ╠═a6fc6796-8375-11eb-02af-01cee06a06dd
# ╟─a6b33828-8375-11eb-027a-130ef7a14128
# ╠═a69ff718-8375-11eb-3965-89ecd2088d97
# ╟─a687f546-8375-11eb-3435-1fa822f71ff1
# ╠═a66ede62-8375-11eb-3746-758cb7efc8a8
# ╟─a657ed74-8375-11eb-1d97-1bd417fa42ee
# ╠═a63e00e4-8375-11eb-10db-897f94a0dc32
# ╟─c54c0456-8373-11eb-1cb6-11bf225bc6b0
# ╠═9ba36afc-83d7-11eb-32d4-4d4d0df107d4
# ╠═cd5eb61e-83d9-11eb-1df3-9389091527e8
# ╟─207ec42e-83df-11eb-129f-b5e34bda1328
# ╠═5a708ed0-83d8-11eb-147c-abf3f80f8c92
# ╠═9b4a0a5e-83d7-11eb-3f02-a5ece0576a9b
# ╠═9aee7716-83d7-11eb-3f3c-d534adaf4bd6
# ╠═baa2d2f0-8373-11eb-27fd-9fc4b6b46360
