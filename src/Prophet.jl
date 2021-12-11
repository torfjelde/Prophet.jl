module Prophet

using Reexport
@reexport using Turing
using LinearAlgebra
using DataFrames
using Distributions.FillArrays

function pandas2dataframe(df)
    columns = map(Symbol, df.columns)
    data = map(columns) do col
        map(identity, getproperty(df, col))
    end
    return DataFrame(Dict(zip(columns, data)))
end

function setup(df, pd, self)
    # history = df[df.y.notnull()].copy()
    history = df.copy()
    self.history_dates = pd.to_datetime(pd.Series(df.ds.unique(), name="ds")).sort_values()
    history = self.setup_dataframe(history, initialize_scales=true)
    self.history = history
    self.set_auto_seasonalities()

    seasonal_features_py, prior_scales, component_cols_py, modes = self.make_all_seasonality_features(history)
    seasonal_features = pandas2dataframe(seasonal_features_py)
    component_cols = pandas2dataframe(component_cols_py)

    # Update some fields.
    self.train_component_cols = component_cols_py
    self.component_modes = modes

    # Determine the changepoints.
    self.set_changepoints()

    dat = (
        y = map(identity, history.y),
        t = map(identity, history.t),
        t_change = self.changepoints_t,
        X = seasonal_features,
        σs = prior_scales,
        τ = self.changepoint_prior_scale,
        s_a = component_cols.additive_terms,
        s_m = component_cols.multiplicative_terms
    )
    return dat
end

function make_changepoint_matrix(t, t_change)
    return make_changepoint_matrix!(zeros(length(t), length(t_change)), t, t_change)
end
function make_changepoint_matrix!(A, t, t_change)
    T = length(t)
    S = length(t_change)

    A .= 0
    row = (similar(A, S) .= 0)
    changepoint_index = 1
    for i = 1:T
        while (changepoint_index ≤ S) && (t[i] ≥ t_change[changepoint_index])
            row[changepoint_index] = 1
            changepoint_index += 1
        end

        A[i, :] .= row
    end

    return A
end

function timeseries_quantiles(xs, q=0.95)
    Δ = (1 - q) / 2
    return quantiles = mapreduce(hcat, xs) do x
        quantile(x, [Δ, 0.5, 1 - Δ])
    end
end

include("model.jl")
include("visualization.jl")

end # module
