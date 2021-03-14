### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 1f62d332-8303-11eb-31e8-4b34a42c4b9f
begin
  using Pkg
    Pkg.activate("MLJ_env", shared=true)

    using DataFrames
    using PlutoUI
    using Test
    using Printf
    using Plots
    using LinearAlgebra
    using Random
    using Statistics
	using StatsBase
	using Distributions
	using Images
	using ImageIO
end

# ╔═╡ f7ad6d8c-8467-11eb-081d-2ba27071eec5
using Colors #, ImageMetadata

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
    nclusters, ndim = length(μ), size(data[1])[1]
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

# ╔═╡ 835e2f8e-8444-11eb-3f3c-d534adaf4bd6
function approx(actᵥ::T, expᵥ::T; ϵ=1e-8) where T <: Real
	abs(actᵥ - expᵥ) ≤ ϵ
end

# ╔═╡ b0655cea-8445-11eb-1fbc-9350beaed253
function approx(actᵥ::AbstractArray{T}, expᵥ::AbstractArray{T};
		ϵ=1e-8) where T <: Real
	all(t -> approx(t[1], t[2]; ϵ), zip(actᵥ, expᵥ))
end

# ╔═╡ 2d4fb59a-8449-11eb-3f02-a5ece0576a9b
function approx(actᵥ::AbstractArray{T}, expᵥ::AbstractArray{T};
		ϵ=1e-8) where {T <: AbstractArray{T1} where T1 <: Real}
	all(t -> approx(t[1], t[2]; ϵ), zip(actᵥ, expᵥ))
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
  # const ϵ = 1e-8
  resp₀ = compute_responsibilities([[1., 2.], [-1., -2.]],
	[0.3, 0.7],                         # weights
    [[0., 0.], [1., 1.]],               # μ
	[[1.5 0.; 0. 2.5], [1. 1.; 1. 2.]]  # σ
  )

  @test size(resp₀) == (2, 2)
  @test approx(resp₀, [0.10512733 0.89487267; 0.46468164 0.53531836])
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
    nclusters, n = length(counts), sum(counts)
    weights = zeros(eltype(counts), nclusters)

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

	@test approx(weights₁, [0.27904865942515705 0.720951340574843])
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
	@test approx(means₂[1], exp_means₂[1])
	@test approx(means₂[2], exp_means₂[2])
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

	exp_covs₂ = [
		[0.60182827 1.20365655; 1.20365655 2.4073131],
		[0.93679654 1.87359307; 1.87359307 3.74718614]
	]

	@test approx(covs₂[1], exp_covs₂[1])
	@test approx(covs₂[2], exp_covs₂[2])
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
	n_iter = 1
    for it ∈ 1:maxiter
		n_iter += 1
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
        (llh_latest - llh) < ϵ && llh_latest > -Base.Inf && break
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
        ## Use sample (from StatsBase) and weights to pick a: 
		## 1 ≤ cluster_id < nclusters
		k = sample(rng, 1:nclusters, Weights(weights), 1; replace=false)[1]

		## Use np.random.multivariate_normal to create data from this cluster
		x = rand(MvNormal(μ[k], σ[k]))  ## ????

		# np.random.multivariate_normal(means[k], covariances[k])
        push!(data, x)
	end
	data
end

# ╔═╡ 00261792-8441-11eb-3969-55f797a05e23


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

# ╔═╡ b3e522d6-844c-11eb-2aa5-750b3ee4c212
md"""
**Checkpoint** for *em* function
"""

# ╔═╡ 9aee7716-83d7-11eb-3f3c-d534adaf4bd6
 begin
 	m_data = [[0.31845368, 5.03182099], [1.45029339, 2.15717573],
 [1.02645431, 4.76042375], [ 5.23493625, -0.81138851], [ 4.65710743, -0.08532351],
 [0.43746553, 4.93778385], [-0.12756373,  5.14296293], [4.18200738, 0.24819247],
 [ 1.65646672, -0.87991369], [-0.69508006,  5.03848976], [6.01348014, 1.87722666],
 [ 0.20099208, -0.93213203], [-0.38268222,  0.26365374], [ 5.03394554, -1.66740986],
 [ 1.53889119, -0.12731715], [0.81448617, 1.63543709], [1.7219243 , 0.28460715],
 [4.90757786, 0.56716088], [0.60512066, 4.87555125], [ 1.30867865, -0.58711045],
 [0.19570654, 5.73836612], [3.97078181, 0.39328234], [0.47190777, 1.47637309],
 [1.32157017, 0.56486368], [3.89988459, 0.92433204], [0.11205488, 6.4072012 ],
 [0.40717315, 1.67134485], [0.56231532, 5.01349864],
 [1.82113515, 0.78892223], [-0.77578232,  4.62650898], [ 0.45032469, -1.01099583],
 [2.25024431, 0.64832921], [1.47157789, 1.68387047], [-0.0420993 ,  0.50923158],
 [1.18348882, 0.56978228], [5.52201954, 0.46015202], [0.69012115, 1.40085221],
 [0.02126064, 4.76228496], [5.32547107, 1.46856369], [2.58605452, 2.11944448],
 [0.182161  , 4.44577711], [ 0.11836846, -0.33361633], [0.66083532, 1.03879141],
 [1.91659334, 0.63739381], [0.28487706, 4.5665366 ], [0.4764496 , 5.72581327],
 [3.10453287, 2.19997423], [-0.85376105,  4.17389402], [0.08542074, 4.38781614],
 [0.21418408, 5.60131432], [-0.29932464,  4.62574683], [2.25835904, 1.1068076 ],
 [ 4.77358263, -0.06002466], [0.73093444, 1.40220082], [0.39250487, 0.57133562],
 [0.8949198, 1.9300788], [-0.25767001,  4.11851619], [0.55042767, 0.21762904],
 [-0.34493741,  4.57098776], [4.71632515, 0.63293438], [1.67820052, 1.29111918],
 [ 1.4862169 , -0.07691204], [5.46625127, 0.01184258], [-1.33846048, -0.20047555],
 [0.01458455, 5.5685916 ], [1.49605116, 4.82940436], [ 5.75963176, -0.13066323],
 [1.70380591, 1.31376129], [ 4.76328545, -0.33344549], [ 0.8945073 , -1.13036243],
 [1.61279831, 0.66792909], [-0.20028537,  5.0349852 ], [-0.79147695,  4.72403614],
 [1.33337808, 1.427949  ], [1.81545643, 2.37052768], [1.02894019, 3.21262401],
 [-0.1914201 ,  4.03020414], [1.45589226, 0.85329139], [2.23603443, 0.87710097],
 [0.32608726, 0.09240739], [-0.18786611,  5.85099597], [4.29931854, 1.35428737],
 [0.49849313, 0.68444377], [-0.8336478 ,  5.81745508], [-0.57795091, -1.6674263 ],
 [1.83292993, 0.7963827 ],[1.01736806, 0.8455941 ], [0.48493064, 0.45675926],
 [-0.47669162,  3.91542167],[0.96494312, 0.65663175], [1.04999151, 0.38350349],
 [1.17458921, 0.73066238],[0.04638356, 4.86196943], [1.66742479, 1.54181187],
 [0.94372389, 1.32117047],[ 5.63522868, -0.0138849 ],[0.52699679, 5.37301739],
 [4.80426637, 0.8017483], [-0.00321541,  0.2083691],[1.38949165, 0.95020545]
	];

 	m_μs = [[0.19570654, 5.73836612], [6.01348014, 1.87722666],
		[0.52699679, 5.37301739]];
	
	m_σs =[[ 3.41361848  -1.91940532; -1.91940532  4.68785076], 
		[ 3.41361848 -1.91940532; -1.91940532  4.68785076], 
		[ 3.41361848 -1.91940532; -1.91940532  4.68785076]];
	
	m_weights = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333];
	
	m_results = em(m_data, m_μs, m_σs, m_weights);
	:done
end

# ╔═╡ 4b50e4b0-8444-11eb-3969-55f797a05e23
begin
	@test approx(m_results[:means], Array{Float64,1}[
			Float64[0.021382851931258, 4.947729003491], 
			Float64[4.942392352282, 0.3136531084758], 
			Float64[1.0818112539095, 0.73903507897]])

	@test approx(m_results[:weights], Float64[0.3007102300634, 
			0.17993710074063, 0.5193526691959])
	
	@test approx(m_results[:covs], Array{Float64,2}[ 
			[0.2932613984949 0.05048454589297; 0.05048454589297 0.352815373349],
			[0.3556437022527 -0.014948748777286; -0.014948748777286 0.6669502500144],
			[0.6711499192647 0.33058964530721; 0.33058964530721 0.9042972427651]])
	
	@test approx(m_results[:weights], [0.3007102300634, 
			0.17993710074063, 0.5193526691959])
	
	@test approx(m_results[:loglik][1:5], [-541.3161249736, -372.1355927523,
			-366.9935696838, -365.6599199088, -364.3348647749, -362.87960462340, 
			-361.3750403053])
end

# ╔═╡ a7135b54-8375-11eb-2d63-392c4fa94aea
md"""
**Now we will fit a mixture of Gaussians** to this data using our implementation of the EM algorithm. As with k-means, it is important to ask how we obtain an initial configuration of mixing weights and component parameters.

In this simple case, we'll take three random points to be the initial cluster means, use the empirical covariance of the data to be the initial covariance in each cluster (a clear overestimate), and set the initial mixing weights to be uniform across clusters.
"""

# ╔═╡ a6fc6796-8375-11eb-02af-01cee06a06dd
begin
	seed₂ = 2010
	Random.seed!(seed₂)
	rng = MersenneTwister(seed₂)

	## Initialization of parameters
	const K = 3
	chosen = sample(rng, 1:length(data), K; replace=false)
  	#        np.random.choice(length(data), K, replace=False)
  	@assert length(chosen) == K

  	init_μ₁ = [data[x] for x in chosen];
  	init_σ₁ = [cov(data) for _ ∈ 1:K]; # cov(data, rowvar=0) => col. is an observ.
  	init_w = [1.0 / K for _ ∈ 1:K];

  	## Run EM
  	results = em(data, init_μ₁, init_σ₁, init_w);
	:done
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

"""

# ╔═╡ 4db52910-844d-11eb-3969-55f797a05e23
results[:means]

# ╔═╡ 6f4476bc-844d-11eb-1fbc-9350beaed253
results[:weights]

# ╔═╡ 5ab06896-844d-11eb-3f3c-d534adaf4bd6
md"""
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
    z = Xμ.^2 ./ σ_x^2 + Yμ.^2 ./ σ_y^2 - 2 * ρ * Xμ .* Yμ ./ (σ_x * σ_y)

  	den = 2. * π * σ_x * σ_y * √(1 - ρ^2)
	exp.(-z / (2. * (1. - ρ^2))) / den
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

# ╔═╡ 5a708ed0-83d8-11eb-147c-abf3f80f8c92
function plot_contours(data, means, covs, title)
    δ = 0.025
    k = length(means)
    X = Y = range(-3.0; stop=9.0, step=δ) 
	
	col = [:green, :red, :indigo]
	parms = []
	scatter([_x[1] for _x ∈ data], [_y[2] for _y ∈ data], legend=false, title=title)
	
	for i ∈ 1:K
		μ, σ = means[i], covs[i]
		σ_x, σ_y = √(σ[1, 1]), √(σ[2, 2])
		σ_xy = σ[1, 2] / (σ_x * σ_y)
		
		push!(parms, (σ_x=σ_x, σ_y=σ_y, μ_x=μ[1], μ_y=μ[2], σ_xy=σ_xy))
	end
  	
	contour!(X, Y, (x, y) -> bivariate_normal(x, y; parms[1]...), color=col[1])
  	contour!(X, Y, (x, y) -> bivariate_normal(x, y; parms[2]...), color=col[2])
  	contour!(X, Y, (x, y) -> bivariate_normal(x, y; parms[3]...), color=col[3])
end

# ╔═╡ 9b4a0a5e-83d7-11eb-3f02-a5ece0576a9b
plot_contours(data, init_μ₁, init_σ₁, "Initial clusters")

# ╔═╡ e9ae52fe-845e-11eb-15c1-65ebc1f4df8e
begin
	# Parameters after running EM to convergence
	plot_contours(data, results[:means], results[:covs], "Final clusters")
end

# ╔═╡ 3ab0e342-845f-11eb-321d-61972f4e6bd4
md"""
Fill in the following code block to visualize the set of parameters we get after running EM for 12 iterations.
"""

# ╔═╡ 4321d874-845f-11eb-107c-c32e7c640947
begin
	res12 = em(data, init_μ₁, init_σ₁, init_w);
	plot_contours(data, res12[:means], res12[:covs], "Clusters after 12 iterations")
end

# ╔═╡ ffd6d3a2-845f-11eb-1308-39662d5cf06e
md"""
**Quiz Question**: Plot the loglikelihood that is observed at each iteration. Is the loglikelihood plot monotonically increasing, monotonically decreasing, or neither [multiple choice]? 
"""

# ╔═╡ a3ac0552-845f-11eb-2283-6defb781f7c7
plot(1:length(results[:loglik]), results[:loglik], lw=3,
	xlabel="Iteration",
	ylabel="Log-likelihood",
	legend=false)

# ╔═╡ 125d76fa-8460-11eb-1d49-57e4eea41e89
md"""
 - The loglikelihood is monotonically increasing
"""

# ╔═╡ c72c03ba-8469-11eb-2fc0-47dab53b1ce9
md"""
## Fitting a Gaussian mixture model for image data¶
"""

# ╔═╡ 075f9b50-8464-11eb-35b3-c945b1e49ccf
begin
	const IMG_DIR = "../../ML_UW_Spec/data/images"
	images = []
	
	for dir_entry ∈ cd(readdir, IMG_DIR)
		println("Found $(dir_entry)")
		sdir = join([IMG_DIR, dir_entry], "/")
		
		for img_file ∈ readdir(sdir)
			push!(images, load(join([sdir, img_file], "/")))
		end
	end
end

# ╔═╡ a27f740e-8465-11eb-171f-3ff85b79e29b
length(images)

# ╔═╡ 77bd32d4-8466-11eb-2713-6b11add92ba3
mosaicview(images[1:12]...; nrow=3, ncol=4)

# ╔═╡ fb990ba8-8466-11eb-13be-7d3f33c0576c
mosaicview(images[400:411]...; nrow=3, ncol=4)

# ╔═╡ 36448e26-8467-11eb-12eb-23daf58ea752
mosaicview(images[700:711]...; nrow=3, ncol=4)

# ╔═╡ d1b3f548-8466-11eb-304e-119b7c99959e
mosaicview(images[1000:1011]...; nrow=3, ncol=4)

# ╔═╡ 8c8995d0-8467-11eb-3efb-d1a7961fdefa
img_data = [
	[mean(red.(img)), mean(green.(img)), mean(blue.(img))] for img ∈ images
]

# ╔═╡ bcc9335c-8469-11eb-1492-d310a071b09a
md"""
We need to come up with initial estimates for the mixture weights and component parameters. Let's take three images to be our initial cluster centers, and let's initialize the covariance matrix of each cluster to be diagonal with each element equal to the sample variance from the full data. As in our test on simulated data, we'll start by assuming each mixture component has equal weight.

This may take a few minutes to run.
"""

# ╔═╡ 7b6cc3d8-846e-11eb-315c-5544ae999cc8
var_col(img_data, i::Integer) = var(x[i] for x ∈ img_data)

# ╔═╡ daffa734-8469-11eb-35ea-376de2c6e1e3
begin
	seed₃ = 2010
	Random.seed!(seed₃)
	rng₃ = MersenneTwister(seed₃)

	# Initalize parameters
	const KC = 4
	const NC = 3  # 3 colors R, G, B
	n_images = length(images)
	
	im_init_μs = [img_data[x] for x ∈ sample(rng₃, 1:n_images, KC; replace=false)]
	@assert length(im_init_μs) == KC
	
	id_mat = Matrix{Float64}(I, NC, NC);
	im_cov = id_mat .* vec([var_col(img_data, c) for c in 1:NC]);
	
	im_init_σs = [im_cov for _ ∈ 1:KC];     # [cov, cov, cov, cov]
	im_init_ws = [1.0 / KC for _ ∈ 1:KC]; # [1/4., 1/4., 1/4., 1/4.]

	## Run our EM algorithm on the image data using the above initializations. 
	## This should converge in about 125 iterations
	res_img = em(img_data, im_init_μs, im_init_σs, im_init_ws);
	:done
end

# ╔═╡ 7fb68656-8471-11eb-3969-55f797a05e23
function _plot(data_x, data_y)
  plot(data_x, data_y, lw=4, xlabel="Iteration", ylabel="Log-likelihood'")
end

# ╔═╡ ab34bece-8471-11eb-3f3c-d534adaf4bd6
begin
	llh_ = res_img[:loglik]
	_plot(1:length(llh_), llh_)
end

# ╔═╡ cccb8bd0-8471-11eb-1fbc-9350beaed253
md"""
The log likelihood increases so quickly on the first few iterations that we can barely see the plotted line. Let's plot the log likelihood after the first three iterations to get a clearer view of what's going on:
"""

# ╔═╡ d63fd518-8471-11eb-147c-abf3f80f8c92
_plot(3:length(llh_), llh_[3:end])

# ╔═╡ d624eb68-8471-11eb-32d4-4d4d0df107d4
md"""
#### Evaluating uncertainty

Next we'll explore the evolution of cluster assignment and uncertainty. Remember that the EM algorithm represents uncertainty about the cluster assignment of each data point through the responsibility matrix. Rather than making a 'hard' assignment of each data point to a single cluster, the algorithm computes the responsibility of each cluster for each data point, where the responsibility corresponds to our certainty that the observation came from that cluster.

We can track the evolution of the responsibilities across iterations to see how these 'soft' cluster assignments change as the algorithm fits the Gaussian mixture model to the data; one good way to do this is to plot the data and color each point according to its cluster responsibilities. Our data are three-dimensional, which can make visualization difficult, so to make things easier we will plot the data using only two dimensions, taking just the [R G], [G B] or [R B] values instead of the full [R G B] measurement for each observation.

"""

# ╔═╡ 37bd479e-8472-11eb-3106-e3d88fb54de4
# TODO...

# ╔═╡ 37a19df0-8472-11eb-2662-6157d71dd17f
md"""
#### Interpreting each cluster

Let's dig into the clusters obtained from our EM implementation. Recall that our goal in this section is to cluster images based on their RGB values. We can evaluate the quality of our clustering by taking a look at a few images that 'belong' to each cluster. We hope to find that the clusters discovered by our EM algorithm correspond to different image categories - in this case, we know that our images came from four categories ('cloudy sky', 'rivers', 'sunsets', and 'trees and forests'), so we would expect to find that each component of our fitted mixture model roughly corresponds to one of these categories.

If we want to examine some example images from each cluster, we first need to consider how we can determine cluster assignments of the images from our algorithm output. This was easy with k-means - every data point had a 'hard' assignment to a single cluster, and all we had to do was find the cluster center closest to the data point of interest. Here, our clusters are described by probability distributions (specifically, Gaussians) rather than single points, and our model maintains some uncertainty about the cluster assignment of each observation.

One way to phrase the question of cluster assignment for mixture models is as follows: how do we calculate the distance of a point from a distribution? Note that simple Euclidean distance might not be appropriate since (non-scaled) Euclidean distance doesn't take direction into account.  For example, if a Gaussian mixture component is very stretched in one direction but narrow in another, then a data point one unit away along the 'stretched' dimension has much higher probability (and so would be thought of as closer) than a data point one unit away along the 'narrow' dimension. 

In fact, the correct distance metric to use in this case is known as [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance). For a Gaussian distribution, this distance is proportional to the square root of the negative log likelihood. This makes sense intuitively - reducing the Mahalanobis distance of an observation from a cluster is equivalent to increasing that observation's probability according to the Gaussian that is used to represent the cluster. This also means that we can find the cluster assignment of an observation by taking the Gaussian component for which that observation scores highest. We'll use this fact to find the top examples that are 'closest' to each cluster.

__Quiz Question:__ Calculate the likelihood (score) of the first image in our data set (`images[0]`) under each Gaussian component through a call to `multivariate_normal.pdf`.  Given these values, what cluster assignment should we make for this image? Hint: don't forget to use the cluster weights.
"""

# ╔═╡ 3787dc3a-8472-11eb-129f-b5e34bda1328
res_img[:weights]

# ╔═╡ 37702d1a-8472-11eb-1df3-9389091527e8
res_img[:means]

# ╔═╡ d606647c-8471-11eb-3f02-a5ece0576a9b
# d = MvNormal(μ[k], σ[k])
#			resp[i, k] = weights[k] * pdf(d, data[i])

# ╔═╡ d92f92e4-8472-11eb-17cf-eb0a1c444ee9
begin
	proba_img = Float64[]
	for k in 1:KC
		d = MvNormal(res_img[:means][k], res_img[:covs][k])
		push!(proba_img, res_img[:weights][k] * pdf(d, img_data[1]))
	end
	proba_img
end

# ╔═╡ d9132afa-8472-11eb-38c5-39d44a1e57f7
md"""
Now we calculate cluster assignments for the entire image dataset using the result of running EM for 60 iterations above.
"""

# ╔═╡ d8f3fd56-8472-11eb-3856-99f30a89c6d3
begin
	p_weights = res_img[:weights]
	p_means = res_img[:means]
	p_covs = res_img[:covs]
	rgb = img_data

	_N = length(img_data)   ## number of images
	_K = length(p_means)    ## number of clusters

	assignments = zeros(Float64, _N)
	probs = zeros(Float64, _N)

	for i ∈ 1:_N
    	## Compute the score of data point i under each Gaussian component:
    	p = [
			p_weights[k] * pdf(MvNormal(p_means[k], p_covs[k]), rgb[i]) for k ∈ 1:_K
    	]
	
    	## Compute assign. of each data point to a given cluster based on the scores:
    	assignments[i] = argmax(p)
    	## For data point i, store the corresponding score under this cluster assign:
    	probs[i] = maximum(p)
	end

	df_assign = DataFrame(:cluster_num => Int.(assignments), :probs => probs, :image => images);
	:done
end

# ╔═╡ 9a3f1a20-8475-11eb-1fbc-9350beaed253
last(df_assign, 5)

# ╔═╡ 9a2134e2-8475-11eb-3f3c-d534adaf4bd6
md"""
We'll use the 'assignment' DataFrame to find the top images from each cluster by sorting the datapoints within each cluster by their score under that cluster (stored in probs). 

Create a function that returns the top 5 images assigned to a given category in our data.
"""

# ╔═╡ 9a00824c-8475-11eb-3969-55f797a05e23
function get_top_images(df, cluster; k=5)
    images_in_cluster = df[df.cluster_num .== cluster, :]
	top_images = sort(images_in_cluster, [:probs], rev=true)[1:k, :]
    top_images.image
end

# ╔═╡ 17ac70be-8476-11eb-3f02-a5ece0576a9b
mosaicview(get_top_images(df_assign, 1)...; nrow=2, ncol=3)

# ╔═╡ 8b254648-8477-11eb-1df3-9389091527e8
mosaicview(get_top_images(df_assign, 2)...; nrow=2, ncol=3)

# ╔═╡ 8b068190-8477-11eb-147c-abf3f80f8c92
mosaicview(get_top_images(df_assign, 3)...; nrow=2, ncol=3)

# ╔═╡ 8ae806ca-8477-11eb-32d4-4d4d0df107d4
mosaicview(get_top_images(df_assign, 4)...; nrow=2, ncol=3)

# ╔═╡ a5dc1b88-8477-11eb-129f-b5e34bda1328
md"""
These look pretty good! Our algorithm seems to have done a good job overall at 'discovering' the four categories that from which our image data was drawn. It seems to have had the most difficulty in distinguishing between rivers and cloudy skies, probably due to the similar color profiles of images in these categories; if we wanted to achieve better performance on distinguishing between these categories, we might need a richer representation of our data than simply the average [R G B] values for each image.
"""

# ╔═╡ Cell order:
# ╟─d67867cc-8302-11eb-3d40-d5a50515cdb9
# ╠═1f62d332-8303-11eb-31e8-4b34a42c4b9f
# ╟─53e17eba-8303-11eb-2471-eb8521bf88d2
# ╟─68c8f63a-8303-11eb-0ae3-c3f39e9acf8d
# ╠═84b9158c-8303-11eb-2ce4-c109ce3ff51a
# ╠═c01ba57e-8303-11eb-34e7-dfb911bac2f2
# ╠═835e2f8e-8444-11eb-3f3c-d534adaf4bd6
# ╠═b0655cea-8445-11eb-1fbc-9350beaed253
# ╠═2d4fb59a-8449-11eb-3f02-a5ece0576a9b
# ╟─580675c0-830a-11eb-16e4-b963a391b138
# ╠═4d69bcec-830c-11eb-0c80-314c8cded6db
# ╠═31835154-830d-11eb-09fd-196046609868
# ╟─430e58a8-8310-11eb-2a53-ed2608f43122
# ╟─42ef055c-8310-11eb-1ae5-d5ae8bf22c1f
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
# ╠═00261792-8441-11eb-3969-55f797a05e23
# ╠═a740cde6-8375-11eb-0f2a-efc1f23f8f38
# ╠═a72ae382-8375-11eb-15fc-c750a071dccf
# ╟─b3e522d6-844c-11eb-2aa5-750b3ee4c212
# ╠═9aee7716-83d7-11eb-3f3c-d534adaf4bd6
# ╠═4b50e4b0-8444-11eb-3969-55f797a05e23
# ╟─a7135b54-8375-11eb-2d63-392c4fa94aea
# ╠═a6fc6796-8375-11eb-02af-01cee06a06dd
# ╟─a6b33828-8375-11eb-027a-130ef7a14128
# ╠═4db52910-844d-11eb-3969-55f797a05e23
# ╠═6f4476bc-844d-11eb-1fbc-9350beaed253
# ╟─5ab06896-844d-11eb-3f3c-d534adaf4bd6
# ╠═a69ff718-8375-11eb-3965-89ecd2088d97
# ╟─a687f546-8375-11eb-3435-1fa822f71ff1
# ╠═a66ede62-8375-11eb-3746-758cb7efc8a8
# ╟─a657ed74-8375-11eb-1d97-1bd417fa42ee
# ╠═a63e00e4-8375-11eb-10db-897f94a0dc32
# ╟─c54c0456-8373-11eb-1cb6-11bf225bc6b0
# ╠═9ba36afc-83d7-11eb-32d4-4d4d0df107d4
# ╠═cd5eb61e-83d9-11eb-1df3-9389091527e8
# ╠═5a708ed0-83d8-11eb-147c-abf3f80f8c92
# ╠═9b4a0a5e-83d7-11eb-3f02-a5ece0576a9b
# ╠═e9ae52fe-845e-11eb-15c1-65ebc1f4df8e
# ╟─3ab0e342-845f-11eb-321d-61972f4e6bd4
# ╠═4321d874-845f-11eb-107c-c32e7c640947
# ╟─ffd6d3a2-845f-11eb-1308-39662d5cf06e
# ╠═a3ac0552-845f-11eb-2283-6defb781f7c7
# ╟─125d76fa-8460-11eb-1d49-57e4eea41e89
# ╟─c72c03ba-8469-11eb-2fc0-47dab53b1ce9
# ╠═075f9b50-8464-11eb-35b3-c945b1e49ccf
# ╠═a27f740e-8465-11eb-171f-3ff85b79e29b
# ╠═77bd32d4-8466-11eb-2713-6b11add92ba3
# ╠═fb990ba8-8466-11eb-13be-7d3f33c0576c
# ╠═36448e26-8467-11eb-12eb-23daf58ea752
# ╠═d1b3f548-8466-11eb-304e-119b7c99959e
# ╠═f7ad6d8c-8467-11eb-081d-2ba27071eec5
# ╠═8c8995d0-8467-11eb-3efb-d1a7961fdefa
# ╟─bcc9335c-8469-11eb-1492-d310a071b09a
# ╠═7b6cc3d8-846e-11eb-315c-5544ae999cc8
# ╠═daffa734-8469-11eb-35ea-376de2c6e1e3
# ╠═7fb68656-8471-11eb-3969-55f797a05e23
# ╠═ab34bece-8471-11eb-3f3c-d534adaf4bd6
# ╟─cccb8bd0-8471-11eb-1fbc-9350beaed253
# ╠═d63fd518-8471-11eb-147c-abf3f80f8c92
# ╟─d624eb68-8471-11eb-32d4-4d4d0df107d4
# ╠═37bd479e-8472-11eb-3106-e3d88fb54de4
# ╟─37a19df0-8472-11eb-2662-6157d71dd17f
# ╠═3787dc3a-8472-11eb-129f-b5e34bda1328
# ╠═37702d1a-8472-11eb-1df3-9389091527e8
# ╠═d606647c-8471-11eb-3f02-a5ece0576a9b
# ╠═d92f92e4-8472-11eb-17cf-eb0a1c444ee9
# ╟─d9132afa-8472-11eb-38c5-39d44a1e57f7
# ╠═d8f3fd56-8472-11eb-3856-99f30a89c6d3
# ╠═9a3f1a20-8475-11eb-1fbc-9350beaed253
# ╟─9a2134e2-8475-11eb-3f3c-d534adaf4bd6
# ╠═9a00824c-8475-11eb-3969-55f797a05e23
# ╠═17ac70be-8476-11eb-3f02-a5ece0576a9b
# ╠═8b254648-8477-11eb-1df3-9389091527e8
# ╠═8b068190-8477-11eb-147c-abf3f80f8c92
# ╠═8ae806ca-8477-11eb-32d4-4d4d0df107d4
# ╟─a5dc1b88-8477-11eb-129f-b5e34bda1328
