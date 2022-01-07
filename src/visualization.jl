using StatsPlots

plot_quantiles(args...; kwargs...) = plot_quantiles!(plot(), args...; kwargs...)
function plot_quantiles!(p::Plots.Plot, xs; q=0.99, kwargs...)
    return plot_quantiles!(p, collect(1:length(xs)), xs; q, kwargs...)
end
function plot_quantiles!(p::Plots.Plot, ts, xs; q=0.99, kwargs...)
    Δ = (1 - q) / 2
    quantiles = mapreduce(hcat, xs) do x
        quantile(x, [Δ, 0.5, 1 - Δ])
    end

    plot!(
        p,
        ts,
        quantiles[2, :];
        ribbon=(quantiles[2, :] - quantiles[1, :], quantiles[3, :] - quantiles[2, :]),
        label="$(q * 100)% credible intervals",
        kwargs...
    )
    return p
end
