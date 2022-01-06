module Prophet

using LinearAlgebra
using Downloads: Downloads

using Reexport
using DataFrames

@reexport using Turing
using Turing.Distributions.FillArrays

# Python stuff.
using Pandas: Pandas
using PyCall: PyCall

# https://github.com/JuliaPy/PyCall.jl/blob/e3e3008ee1b3e449c6382293f58a1f85384beea6/README.md#using-pycall-from-julia-modules
const pd = PyCall.PyNULL()
const pyprophet = PyCall.PyNULL()

function __init__()
    copy!(pd, PyCall.pyimport_conda("pandas", "pandas"))
    copy!(pyprophet, PyCall.pyimport_conda("prophet", "prophet"))
end

function pandas2dataframe(df)
    columns = map(Symbol, df.columns)
    data = map(columns) do col
        map(identity, getproperty(df, col))
    end
    return DataFrame(Dict(zip(columns, data)))
end

setup(df::DataFrame) = setup(Pandas.DataFrame(df).pyo)
function setup(df::PyCall.PyObject)
    # Ensure that we're working with a dataframe.
    isinstance = PyCall.pybuiltin("isinstance")
    isinstance(df, pd.DataFrame) || throw(ArgumentError("`df` must be of type `PyObject <class 'pandas.core.frame.DataFrame'>`"))

    return setup(df, pyprophet.Prophet())
end
function setup(df, self)
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

    # Extract the data we're interested in.
    t = map(identity, history.t)
    X = Matrix(seasonal_features)
    y = map(identity, history.y)
    t_change = self.changepoints_t
    s_a = component_cols.additive_terms
    s_m = component_cols.multiplicative_terms
    σs = prior_scales
    τ = self.changepoint_prior_scale

    A = make_changepoint_matrix(t, t_change)
    return (; t, X, y, A, t_change, s_a, s_m, τ, σs)
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

function load_peyton_manning_dataset()
    path = Downloads.download(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    return DataFrame(CSV.File(path))
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
