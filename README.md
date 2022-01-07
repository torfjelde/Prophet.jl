# `Prophet.jl`

A quick and dirty approach to get [`Turing.jl`](https://github.com/TuringLang/Turing.jl) running inference similar to [`prophet`](https://github.com/facebook/prophet/).

## Quickstart

``` julia
using Prophet

# Load an example dataset.
df = Prophet.load_peyton_manning();

# Instantiate the model.
model = prophet(df);

# [Optional] For much improved performance.
# using ReverseDiff, Memoization
# Turing.setadbackend(:reversediff)
# Turing.setrdcache(true)

# Inference.
chain = sample(model, NUTS(500, 0.8), 500);

# Predictions.
predictions_chain = predict(decondition(model), chain);
predictions_matrix = Matrix(predictions_chain);

# Visualize predictions.
Prophet.plot_quantiles(model.args.t, eachcol(predictions_matrix))
```

