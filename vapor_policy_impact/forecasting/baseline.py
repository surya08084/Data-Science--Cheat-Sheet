"""No-policy baseline forecaster: Scenario B (methodology section 5, 11).

Two independent forecasters are combined into an ensemble:

* :class:`BSTSBaselineForecaster` -- a structural time-series (local level +
  seasonal + donor-pool exogenous regressor) state-space model fit with
  statsmodels' ``UnobservedComponents``. This is the frequentist (Kalman
  filter / MLE) sibling of a Bayesian Structural Time Series model -- same
  model class as CausalImpact, same posterior-style output via
  ``.simulate()`` sample paths, without a full MCMC treatment of parameter
  uncertainty (documented; extending to a fully Bayesian fit is a drop-in
  swap of the fitting routine, not the model).
* :class:`QuantileGBMForecaster` -- a pooled, donor-state-trained direct
  multi-horizon quantile-regression forecaster (one gradient-boosted
  quantile model per horizon), for when richer engineered features (price,
  promo, distribution, momentum) matter more than the state-space model can
  capture.

Both forecasters use the *donor pool*, not just the target state's own
history: the BSTS model's exogenous regressor is the Layer-2 synthetic
control series (evaluated on donors' real, currently-observed data for the
forecast window); the quantile GBM model is trained on pooled donor-state
rows so it has enough data to learn nonlinear feature effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.structural import UnobservedComponents


@dataclass
class BaselineForecastResult:
    weeks: np.ndarray
    draws: np.ndarray  # n_draws x horizon, log-volume scale
    mean: np.ndarray
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    source: str


def _summarize_draws(weeks: np.ndarray, draws: np.ndarray, source: str) -> BaselineForecastResult:
    return BaselineForecastResult(
        weeks=weeks,
        draws=draws,
        mean=draws.mean(axis=0),
        q10=np.quantile(draws, 0.10, axis=0),
        q50=np.quantile(draws, 0.50, axis=0),
        q90=np.quantile(draws, 0.90, axis=0),
        source=source,
    )


class BSTSBaselineForecaster:
    def __init__(
        self,
        seasonal_period: float = 52.0,
        harmonics: int = 2,
        n_draws: int = 500,
        random_state: int = 0,
        level: str = "local level",
    ):
        self.seasonal_period = seasonal_period
        self.harmonics = harmonics
        self.n_draws = n_draws
        self.random_state = random_state
        self.level = level

    def fit_and_forecast(
        self,
        target_log_volume: pd.Series,
        synthetic_exog: pd.Series | None,
        horizon: int = 13,
    ) -> BaselineForecastResult:
        history = target_log_volume.dropna().sort_index()
        if len(history) < 2 * self.seasonal_period:
            # not enough history for a full seasonal cycle-and-a-half: drop seasonality
            freq_seasonal = None
        else:
            freq_seasonal = [{"period": self.seasonal_period, "harmonics": self.harmonics}]

        last_week = int(history.index.max())
        forecast_weeks = np.arange(last_week + 1, last_week + 1 + horizon)

        exog_hist = exog_future = None
        if synthetic_exog is not None:
            exog_hist = synthetic_exog.reindex(history.index).to_numpy().reshape(-1, 1)
            exog_future = synthetic_exog.reindex(forecast_weeks).to_numpy().reshape(-1, 1)
            if np.isnan(exog_hist).any() or np.isnan(exog_future).any():
                # donor coverage doesn't fully span history/forecast window -> drop exog
                exog_hist = exog_future = None

        model = UnobservedComponents(
            endog=history.to_numpy(),
            level=self.level,
            freq_seasonal=freq_seasonal,
            exog=exog_hist,
        )
        res = model.fit(disp=False)

        sims = res.simulate(
            nsimulations=horizon,
            repetitions=self.n_draws,
            anchor="end",
            exog=exog_future,
            random_state=self.random_state,
        )
        draws = np.asarray(sims)
        if draws.ndim == 3:
            draws = draws[:, 0, :]
        draws = draws.T  # -> n_draws x horizon

        return _summarize_draws(forecast_weeks, draws, source="bsts")


def _build_direct_horizon_training_set(
    feature_panel: pd.DataFrame,
    states: list[str],
    horizon: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = feature_panel[feature_panel["state"].isin(states)].sort_values(["state", "week"]).copy()
    if df.duplicated(subset=["state", "week"]).any():
        raise ValueError(
            "feature_panel must have exactly one row per (state, week) -- e.g. built via "
            "features.engineering.build_aggregated_feature_series -- otherwise groupby('state').shift(k) "
            "shifts by row position instead of by calendar week and silently misaligns every lag/target."
        )
    df["target"] = df.groupby("state")["log_volume"].shift(-horizon)
    cols = ["state", "week", "target"] + feature_cols
    return df[cols].dropna()


class QuantileGBMForecaster:
    DEFAULT_FEATURES = [
        "volume_lag_1",
        "volume_lag_4",
        "volume_lag_13",
        "volume_lag_52",
        "rolling_mean_13",
        "rolling_std_13",
        "momentum_4v4",
        "fourier_sin_1",
        "fourier_cos_1",
        "fourier_sin_2",
        "fourier_cos_2",
        "price",
        "promo_depth",
        "distribution_acv",
    ]

    def __init__(self, quantiles: tuple[float, ...] = (0.10, 0.50, 0.90), random_state: int = 0):
        self.quantiles = quantiles
        self.random_state = random_state

    def fit_and_forecast(
        self,
        feature_panel: pd.DataFrame,
        donor_states: list[str],
        target_state: str,
        horizon: int = 13,
        feature_cols: list[str] | None = None,
    ) -> BaselineForecastResult:
        feature_cols = feature_cols or self.DEFAULT_FEATURES
        train_pool_states = list(dict.fromkeys(donor_states + [target_state]))

        target_rows = feature_panel[feature_panel["state"] == target_state].sort_values("week")
        valid_target_rows = target_rows.dropna(subset=["log_volume"])
        if valid_target_rows.empty:
            raise ValueError(f"No log_volume history available for target_state={target_state!r}.")
        # Use the most recent row with a real outcome as the forecast cutoff, but don't
        # require every long-lag feature (e.g. volume_lag_52) to be non-null there: a
        # state with less than a year of pre-period history will never satisfy that, which
        # would otherwise make the cutoff row unselectable. Missing features on just the
        # cutoff row are imputed with the pooled donor/target training median instead.
        cutoff_row = valid_target_rows.iloc[-1]
        cutoff_week = int(cutoff_row["week"])
        forecast_weeks = np.arange(cutoff_week + 1, cutoff_week + 1 + horizon)

        pool_medians = feature_panel[feature_panel["state"].isin(train_pool_states)][feature_cols].median()
        X_pred = cutoff_row[feature_cols].fillna(pool_medians).to_numpy(dtype=float).reshape(1, -1)

        preds = {q: np.zeros(horizon) for q in self.quantiles}
        for h in range(1, horizon + 1):
            train = _build_direct_horizon_training_set(feature_panel, train_pool_states, h, feature_cols)
            if len(train) < 30:
                # too little pooled data for this horizon -> fall back to last known level
                for q in self.quantiles:
                    preds[q][h - 1] = cutoff_row["log_volume"]
                continue
            X = train[feature_cols].to_numpy(dtype=float)
            y = train["target"].to_numpy(dtype=float)
            for q in self.quantiles:
                gbm = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=150,
                    max_depth=2,
                    learning_rate=0.08,
                    subsample=0.8,
                    random_state=self.random_state,
                )
                gbm.fit(X, y)
                preds[q][h - 1] = gbm.predict(X_pred)[0]

        # build pseudo-draws from the quantile predictions for a consistent downstream
        # interface (Monte Carlo combiner expects draws, not just 3 quantile curves):
        # sample per-horizon from a triangular-ish distribution implied by q10/q50/q90.
        rng = np.random.default_rng(self.random_state)
        n_draws = 500
        q10, q90 = preds[self.quantiles[0]], preds[self.quantiles[-1]]
        q50 = preds.get(0.5, (q10 + q90) / 2)
        draws = np.empty((n_draws, horizon))
        for h in range(horizon):
            left_scale = max((q50[h] - q10[h]), 1e-4)
            right_scale = max((q90[h] - q50[h]), 1e-4)
            u = rng.random(n_draws)
            left = u < 0.5
            draws[left, h] = q50[h] - np.abs(rng.normal(0, left_scale, left.sum()))
            draws[~left, h] = q50[h] + np.abs(rng.normal(0, right_scale, (~left).sum()))

        return _summarize_draws(forecast_weeks, draws, source="quantile_gbm")


def ensemble_baseline_forecast(
    results: list[BaselineForecastResult],
    weights: list[float] | None = None,
) -> BaselineForecastResult:
    """Pool draws from multiple baseline forecasters (weighted by e.g. LOSO
    backtest performance per category, section 11) into a single draw set.
    """
    if weights is None:
        weights = [1.0] * len(results)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    weeks = results[0].weeks
    n_draws_total = 2000
    pooled = []
    for res, w in zip(results, weights):
        n_take = int(round(n_draws_total * w))
        idx = np.random.default_rng(0).choice(res.draws.shape[0], size=n_take, replace=True)
        pooled.append(res.draws[idx])
    draws = np.concatenate(pooled, axis=0)
    return _summarize_draws(weeks, draws, source="ensemble")
