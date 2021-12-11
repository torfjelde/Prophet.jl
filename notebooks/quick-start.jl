### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 6d616398-aceb-42dd-b770-6caf775e3278
using Pkg; Pkg.activate("..")

# ╔═╡ e2faa7b8-150e-4dd9-9e43-80752d4d928f
using Revise

# ╔═╡ d8e603ec-38b4-4256-8e70-631d8dad679e
using Prophet

# ╔═╡ e5cec0e1-dca9-434e-b9b9-4ad95a3797c1
using PyCall

# ╔═╡ 647c06b9-7080-4141-b8b3-1ed3384c12f8
using ReverseDiff, Memoization

# ╔═╡ 388a810f-6997-4ee9-88da-f38643213b3f
md"# Prophet"

# ╔═╡ fb6cf509-c64c-4075-a806-c7c7d814fbe5
md"""
Under the hood, Prophet is really just an **additive regression model** with four main components:
- A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
- A yearly seasonal component modeled using Fourier series.
- A weekly seasonaly component using dummy variables.
- A user-provided list of important holidays.
"""

# ╔═╡ 1bd16381-5b80-4490-b525-d5d2eff2fdd3
md"# Setup"

# ╔═╡ d0adf2be-57d5-11ec-3813-5955134c7918
pwd()

# ╔═╡ 13b8da93-9a47-4774-9bcb-33760d20d103
pd = pyimport("pandas")

# ╔═╡ 2d3bd93f-3842-4dce-88f3-4bd0a2874de4
pyprophet = pyimport("prophet")

# ╔═╡ bcfba092-1228-4d19-b9ce-536009fd46af
df = pd.read_csv("../examples/example_wp_log_peyton_manning.csv")

# ╔═╡ 0a6a7419-0997-40cc-9e84-9ca92c0f7bc5
data = Prophet.setup(df, pd, pyprophet.Prophet())

# ╔═╡ 496070c0-cdf7-4d0c-a5d0-4255a72670f0
md"# Setup model"

# ╔═╡ d547d789-1c19-4ac2-b77d-299c344e73df
args = Prophet.setup(Prophet.prophet, data);

# ╔═╡ 590e1f4d-3138-4081-90f1-5d19d9b76a8c
model = Prophet.prophet(args...)

# ╔═╡ c683a36a-432f-4884-ac43-098b80d1bcdb
model()

# ╔═╡ 03e00385-cae0-4f0c-a862-0ccfd2962b00
md"# Inference"

# ╔═╡ 573847a4-3f6b-4ddc-8611-1306e2d39d91
Turing.setadbackend(:reversediff)

# ╔═╡ d38b2c65-e123-4f7d-8b22-b7977b61a9e2
Turing.setrdcache(true)

# ╔═╡ 16213d02-3ba0-453e-ae00-67feba650833
chain = sample(model, NUTS(500, 0.8), 500);

# ╔═╡ 283ec076-8361-4878-b831-ad8f31d03781
chain

# ╔═╡ f438c753-810a-4017-b758-7aea53c4ce08
chain_multithreaded = sample(model, NUTS(), MCMCThreads(), 1000, 4);

# ╔═╡ 50eb53cf-f529-4979-bea4-91d11df0fc2e
addprocs

# ╔═╡ 0239caca-cd3c-47db-ba40-fc448ba88c5f
md"# Prior"

# ╔═╡ 6edac959-ba71-46fe-a703-49efbeec5c11
using StatsPlots

# ╔═╡ 13518c8d-dda0-401b-8fed-65a7575a609c
plot(model())

# ╔═╡ c950db70-2f64-4603-ad7f-6e013848648c
md"# Predictions"

# ╔═╡ b7e98ef7-e617-4142-9441-580f93baddda
predictions_chain = predict(Prophet.prophet(merge(args, (y = missing, ))...), chain);

# ╔═╡ b79b316e-5f86-44f8-8c5c-f62efb4b5f52
predictions = Array(predictions_chain)

# ╔═╡ a42b1765-a5ee-4fc4-ba66-472a0879fce0
begin
	p = Prophet.plot_quantiles(data.t, eachcol(predictions))
	scatter!(p, data.t, data.y, color=:black, markersize=2, label="true")
end

# ╔═╡ adfd84ad-4856-47ae-aa68-72bea5e08b2d
modelpy = pyprophet.Prophet()

# ╔═╡ cc571eea-e782-4f36-82b9-db1e691727a4
modelpy.fit(df)

# ╔═╡ f29a4011-c8e6-45b7-8898-14ec3ec762cc
modelpy.make_future_dataframe(periods=10)

# ╔═╡ c18e184c-2594-4687-97b4-3c275d28a4a5
modelpy.seasonalities

# ╔═╡ 50c58bb3-6f32-4cc2-a529-beb5ac45f168
md"# Outlier detection"

# ╔═╡ 0e476ab5-7569-4183-9650-52e98509aa66
qs = Prophet.timeseries_quantiles(eachcol(predictions), 0.99)

# ╔═╡ aa0a77b6-1c41-42d7-90b7-17b02f66702e
outlier_mask = (data.y .< qs[1, :]) .| (data.y .> qs[3, :])

# ╔═╡ c4468a79-330f-43d3-840d-03a5e829860f
begin
	Prophet.plot_quantiles(data.t, eachcol(predictions), q=0.99)
	scatter!(
		data.t[(!).(outlier_mask)], 
		data.y[(!).(outlier_mask)], 
		markersize=2, color=:black, label=""
	)
	scatter!(
		data.t[outlier_mask], 
		data.y[outlier_mask], 
		markersize=2, color=:red, label="Outliers"
	)
end

# ╔═╡ 3ca8693c-9ec9-42c3-8c6b-53955dd9cd99
md"# Python"

# ╔═╡ bc9869bc-0372-4172-9839-a2556f1cd6a0
pymodel = pyprophet.Prophet(mcmc_samples=1000)

# ╔═╡ 4170d781-32ec-478b-be7a-11fec20aae0a
stan_fit = pymodel.fit(df)

# ╔═╡ Cell order:
# ╟─388a810f-6997-4ee9-88da-f38643213b3f
# ╠═fb6cf509-c64c-4075-a806-c7c7d814fbe5
# ╠═1bd16381-5b80-4490-b525-d5d2eff2fdd3
# ╠═d0adf2be-57d5-11ec-3813-5955134c7918
# ╠═6d616398-aceb-42dd-b770-6caf775e3278
# ╠═e2faa7b8-150e-4dd9-9e43-80752d4d928f
# ╠═d8e603ec-38b4-4256-8e70-631d8dad679e
# ╠═e5cec0e1-dca9-434e-b9b9-4ad95a3797c1
# ╠═13b8da93-9a47-4774-9bcb-33760d20d103
# ╠═2d3bd93f-3842-4dce-88f3-4bd0a2874de4
# ╠═bcfba092-1228-4d19-b9ce-536009fd46af
# ╠═0a6a7419-0997-40cc-9e84-9ca92c0f7bc5
# ╠═496070c0-cdf7-4d0c-a5d0-4255a72670f0
# ╠═d547d789-1c19-4ac2-b77d-299c344e73df
# ╠═590e1f4d-3138-4081-90f1-5d19d9b76a8c
# ╠═c683a36a-432f-4884-ac43-098b80d1bcdb
# ╠═03e00385-cae0-4f0c-a862-0ccfd2962b00
# ╠═647c06b9-7080-4141-b8b3-1ed3384c12f8
# ╠═573847a4-3f6b-4ddc-8611-1306e2d39d91
# ╠═d38b2c65-e123-4f7d-8b22-b7977b61a9e2
# ╠═16213d02-3ba0-453e-ae00-67feba650833
# ╠═283ec076-8361-4878-b831-ad8f31d03781
# ╠═f438c753-810a-4017-b758-7aea53c4ce08
# ╠═50eb53cf-f529-4979-bea4-91d11df0fc2e
# ╠═0239caca-cd3c-47db-ba40-fc448ba88c5f
# ╠═6edac959-ba71-46fe-a703-49efbeec5c11
# ╠═13518c8d-dda0-401b-8fed-65a7575a609c
# ╠═c950db70-2f64-4603-ad7f-6e013848648c
# ╠═b7e98ef7-e617-4142-9441-580f93baddda
# ╠═b79b316e-5f86-44f8-8c5c-f62efb4b5f52
# ╠═a42b1765-a5ee-4fc4-ba66-472a0879fce0
# ╠═adfd84ad-4856-47ae-aa68-72bea5e08b2d
# ╠═cc571eea-e782-4f36-82b9-db1e691727a4
# ╠═f29a4011-c8e6-45b7-8898-14ec3ec762cc
# ╠═c18e184c-2594-4687-97b4-3c275d28a4a5
# ╠═50c58bb3-6f32-4cc2-a529-beb5ac45f168
# ╠═0e476ab5-7569-4183-9650-52e98509aa66
# ╠═aa0a77b6-1c41-42d7-90b7-17b02f66702e
# ╠═c4468a79-330f-43d3-840d-03a5e829860f
# ╠═3ca8693c-9ec9-42c3-8c6b-53955dd9cd99
# ╠═bc9869bc-0372-4172-9839-a2556f1cd6a0
# ╠═4170d781-32ec-478b-be7a-11fec20aae0a
