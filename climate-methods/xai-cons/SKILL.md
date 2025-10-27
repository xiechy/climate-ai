Purpose: Apply an emergent constraint (EC) to reduce uncertainty of a future target  using an observed predictor  and a multi-model ensemble (MME). Provide posterior  distribution and standard EC skill metrics.

Inputs

CSV with columns: model,X,Y[,weight] (one row per model). Scalars per model.

Observational estimate: X_obs and optional standard deviation X_obs_sd.

Options: leave-one-out validation, bootstrap samples, output directory.

Outputs

results.json: correlation, regression params, posterior mean/SD/quantiles, variance reduction, CRPSS, spread/error ratio, LOO scores.

posterior_samples.npy (optional): draws of  given .

fig_* (optional): quick-look plots if enabled.

Procedure

Validate inputs and units; drop NaNs; apply optional weights.

Fit linear EC: . Report  with p-value.

Posterior for : sample regression parameter uncertainty and observational uncertainty (if X_obs_sd given), propagate to  with bootstrap.

Metrics

Prior vs posterior variance and 66%/90% CI shrinkage.

Relative reduced variance RRV = .

Perfect-model LOO: RMSE, CRPS, CRPSS, spread/error ratio.

Save artifacts.