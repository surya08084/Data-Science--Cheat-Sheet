"""Layer 3: heterogeneity / effect-transport model (methodology section 4-5, 15).

Learns how the Layer-1 treatment effect varies with state and policy
characteristics, so a magnitude can be *predicted* for a brand-new state that
has no post-treatment data of its own. With only 8-9 treated states, a
regularized linear meta-regression (Ridge, with leave-one-out selection of
the penalty) is the defensible choice over a data-hungry causal forest -- the
methodology doc calls this out explicitly in section 4's recommendation. A
`model_type="random_forest"` option is included for when more treated states
accumulate and a nonlinear CATE model becomes viable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


@dataclass
class EffectTransportModel:
    model_type: str = "ridge"
    _scaler: StandardScaler | None = None
    _model=None
    _feature_names: list | None = None
    _train_mean_effect: float | None = None

    def fit(self, cohort_effects: pd.Series, covariates: pd.DataFrame) -> "EffectTransportModel":
        """`cohort_effects`: state -> scalar cumulative effect (e.g. mean ATT over e=0..12).
        `covariates`: state-indexed numeric feature table (state characteristics + policy
        characteristics, already one-hot/numeric-encoded).
        """
        aligned = covariates.loc[cohort_effects.index]
        self._feature_names = list(aligned.columns)
        X = aligned.to_numpy(dtype=float)
        y = cohort_effects.to_numpy(dtype=float)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        if self.model_type == "random_forest":
            self._model = RandomForestRegressor(
                n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0
            ).fit(Xs, y)
        else:
            # cv=None triggers sklearn's efficient built-in leave-one-out (via the
            # generalized-CV formula) -- appropriate given only 8-9 treated states,
            # and avoids the degenerate per-fold R^2 you'd get from explicit KFold(n)
            # where every test fold has a single point.
            self._model = RidgeCV(alphas=np.logspace(-2, 3, 25), cv=None).fit(Xs, y)

        self._train_mean_effect = float(np.mean(y))
        return self

    def predict(self, covariates_row: pd.Series) -> float:
        if self._model is None:
            raise RuntimeError("Call .fit() before .predict().")
        x = covariates_row[self._feature_names].to_numpy(dtype=float).reshape(1, -1)
        xs = self._scaler.transform(x)
        return float(self._model.predict(xs)[0])

    def transport_scale(self, pooled_curve: pd.Series, covariates_row: pd.Series) -> float:
        """Ratio between this state's predicted cumulative effect and the pooled
        (all-cohort) cumulative effect over the same post-period window. Applying
        this single scale to both the point pooled curve *and* every Layer-1
        bootstrap replicate curve (see scenario.combiner) is what lets the
        transported effect carry Layer-1's uncertainty forward, without requiring
        this small-N meta-model to produce its own well-calibrated interval.
        """
        predicted = self.predict(covariates_row)
        pooled_post = pooled_curve[pooled_curve.index >= 0]
        pooled_summary = float(pooled_post.mean()) if len(pooled_post) else float(pooled_curve.mean())
        return 1.0 if abs(pooled_summary) < 1e-9 else predicted / pooled_summary

    def transport_curve(self, pooled_curve: pd.Series, covariates_row: pd.Series) -> pd.Series:
        """Rescale the Layer-1 pooled event-time ATT(e) curve to a new state's predicted
        magnitude: shape follows the pooled dynamics, magnitude follows this model's
        state-specific prediction (section 5, Step B).
        """
        return pooled_curve * self.transport_scale(pooled_curve, covariates_row)
