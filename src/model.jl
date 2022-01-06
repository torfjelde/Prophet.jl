function linear_trend(k, m, δ, t, A, t_change)
    return (k .+ A * δ) .* t .+ (m .+ A * (-t_change .* δ))
end

@model function DoubleExponential(μ, λ)
    # NOTE: Parameterization in Stan is the inverse, i.e. our `Exponential(λ)`
    # is equivalent to Stan's `Exponential(1 / λ)`.
    α ~ Exponential(λ)
    δ ~ MvNormal(μ, α * I)
end

"""
    prophet(t, X, y, A, t_change, s_a, s_m, τ, σs)

# Arguments
- `t::AbstractVector`: time interval, i.e. monotonically increasing sequence with `t[1] = 0` and `t[end] = 1`.
- `X::AbstractMatrix`: represents the features 
- `y::AbstractVector`:
- `A`: changepoint matrix.
- `t_change`
- `s_a`: indicator of additive features.
- `s_m`: indicator of multiplicative features.

"""
@model function prophet(t, X, y, A, t_change, s_a, s_m, τ, σs)
    # //priors
    # k ~ normal(0, 5);
    # m ~ normal(0, 5);
    # delta ~ double_exponential(0, tau);
    # sigma_obs ~ normal(0, 0.5);
    # beta ~ normal(0, sigmas);

    # Number of changepoints.
    S = length(t_change)

    # Base trend growth rate.
    k ~ Normal(0, 5)
    # Trend offset.
    m ~ Normal(0, 5)
    # Trend rate adjustments.
    δ = @submodel DoubleExponential(Zeros{eltype(k)}(S), τ)

    σ_obs ~ truncated(Normal(0, 0.5), 0, Inf)
    β ~ MvNormal(Diagonal(σs))

    trend = linear_trend(k, m, δ, t, A, t_change)

    # // Likelihood
    # y ~ normal(
    #   trend
    #     .* (1 + X * (beta .* s_m))
    #     + X * (beta .* s_a),
    #     sigma_obs
    #   );
    y ~ MvNormal(
        trend .* (1 .+ X * (β .* s_m)) + X * (β .* s_a),
        σ_obs * I
    )
end

function setup(::typeof(prophet), data)
    X = Matrix(data.X)
    A = similar(X, length(data.t), length(data.t_change))
    make_changepoint_matrix!(A, data.t, data.t_change)

    return (
        t = data.t,
        X = X,
        y = data.y,
        A = A,
        t_change = data.t_change,
        s_a = data.s_a,
        s_m = data.s_m,
        τ = data.τ,
        σs = data.σs
    )
end
