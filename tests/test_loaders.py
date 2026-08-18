import numpy as np
import pandas as pd
import pytest

from vapor_policy_impact.data.loaders import (
    build_fallback_state_covariates,
    date_to_week_index,
    infer_week_zero_date,
    load_policy_calendar,
    load_real_panel,
    load_state_covariates,
)


@pytest.fixture
def fake_raw_data():
    dates = pd.date_range("2023-01-02", periods=20, freq="W-MON")  # Mondays
    rng = np.random.default_rng(0)
    rows = []
    for state in ["CA", "TX", "NY"]:
        for cat in ["Vapor", "Cigarettes"]:
            for mfg in ["ALTRIA_CO", "OTHER_MFG"]:
                for d in dates:
                    rows.append(
                        {
                            "STATE_CD": state,
                            "WEEK_END_DATE": d,
                            "CATEGORY_DESC": cat,
                            "MFG_GROUP": mfg,
                            "UNIT_VOLUME": rng.uniform(10, 100),
                            "AVG_PRICE": rng.uniform(5, 8),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def col_map():
    return {
        "state": "STATE_CD",
        "week": "WEEK_END_DATE",
        "category": "CATEGORY_DESC",
        "manufacturer_group": "MFG_GROUP",
        "volume": "UNIT_VOLUME",
        "price": "AVG_PRICE",
    }


def test_infer_week_zero_date_floors_to_monday():
    dates = pd.Series(pd.to_datetime(["2023-01-05", "2023-01-10", "2023-02-01"]))  # Thu, Tue, Wed
    wz = infer_week_zero_date(dates)
    assert wz.weekday() == 0  # Monday
    assert wz <= dates.min()


def test_date_to_week_index_basic():
    wz = pd.Timestamp("2023-01-02")
    assert date_to_week_index("2023-01-02", wz) == 0
    assert date_to_week_index("2023-01-09", wz) == 1
    assert date_to_week_index("2023-04-10", wz) == 14


def test_load_real_panel_renames_and_converts_dates(fake_raw_data, col_map):
    panel, week_zero = load_real_panel(fake_raw_data, col_map)
    assert week_zero == pd.Timestamp("2023-01-02")
    assert list(panel["week"].sort_values().unique())[:3] == [0, 1, 2]
    assert set(panel["category"]) == {"Vapor", "Cigarettes"}
    assert set(panel["manufacturer_group"]) == {"ALTRIA_CO", "OTHER_MFG"}
    # optional columns not in the mapping should be filled, not missing
    assert "distribution_acv" in panel.columns
    assert panel["distribution_acv"].isna().all()
    # optional columns not mapped default sku/manufacturer/brand to manufacturer_group
    assert set(panel["sku"]) == set(panel["manufacturer_group"])


def test_load_real_panel_matches_pandasschema_columns(fake_raw_data, col_map):
    from vapor_policy_impact.config import PanelSchema

    panel, _ = load_real_panel(fake_raw_data, col_map)
    assert list(panel.columns) == list(PanelSchema().raw_columns)


def test_load_real_panel_rejects_missing_required_column(fake_raw_data):
    bad_map = {"state": "STATE_CD", "week": "WEEK_END_DATE", "category": "CATEGORY_DESC"}  # missing manufacturer_group, volume
    with pytest.raises(ValueError, match="missing required"):
        load_real_panel(fake_raw_data, bad_map)


def test_load_real_panel_rejects_unresolved_source_column(fake_raw_data):
    bad_map = {
        "state": "STATE_CD",
        "week": "WEEK_END_DATE",
        "category": "CATEGORY_DESC",
        "manufacturer_group": "MFG_GROUP",
        "volume": "NOT_A_REAL_COLUMN",
    }
    with pytest.raises(ValueError, match="not present in the source data"):
        load_real_panel(fake_raw_data, bad_map)


def test_load_real_panel_week_index_passthrough_when_not_a_date():
    df = pd.DataFrame({"S": ["A", "A"], "W": [5, 6], "C": ["Vapor", "Vapor"], "M": ["Altria", "Altria"], "V": [1.0, 2.0]})
    col_map = {"state": "S", "week": "W", "category": "C", "manufacturer_group": "M", "volume": "V"}
    panel, week_zero = load_real_panel(df, col_map, week_is_date=False)
    assert week_zero is None
    assert list(panel["week"]) == [5, 6]


def test_load_policy_calendar_from_dict_matches_panel_week_zero(fake_raw_data, col_map):
    panel, week_zero = load_real_panel(fake_raw_data, col_map)
    calendar = load_policy_calendar({"CA": "2023-04-10"}, week_zero_date=week_zero)
    assert calendar.treated_states == {"CA": 14}
    assert calendar.is_treated("CA")
    assert not calendar.is_treated("TX")


def test_load_policy_calendar_from_dataframe():
    wz = pd.Timestamp("2023-01-02")
    df = pd.DataFrame({"st": ["CA", "TX"], "dt": ["2023-01-16", "2023-01-23"]})
    calendar = load_policy_calendar(df, week_zero_date=wz, state_column="st", date_column="dt")
    assert calendar.treated_states == {"CA": 2, "TX": 3}


def test_build_fallback_state_covariates_computes_vapor_share(fake_raw_data, col_map):
    panel, _ = load_real_panel(fake_raw_data, col_map)
    with pytest.warns(UserWarning, match="No state covariates supplied"):
        cov = build_fallback_state_covariates(panel)
    assert set(cov["state"]) == {"CA", "TX", "NY"}
    # two categories with roughly equal random volume -> vapor share should be near 0.5
    assert (cov["baseline_vapor_share"] > 0.3).all() and (cov["baseline_vapor_share"] < 0.7).all()
    for col in ["population_m", "urbanization_pct", "median_income_k", "border_state", "retail_density"]:
        assert (cov[col] == 0.0).all()


def test_load_state_covariates_falls_back_when_source_none(fake_raw_data, col_map):
    panel, _ = load_real_panel(fake_raw_data, col_map)
    with pytest.warns(UserWarning):
        cov = load_state_covariates(None, panel["state"].unique(), panel_for_fallback=panel)
    assert len(cov) == 3


def test_load_state_covariates_maps_external_source():
    external = pd.DataFrame(
        {
            "st": ["CA", "TX"],
            "pop": [39.0, 30.0],
            "urban_pct": [95.0, 84.0],
        }
    )
    mapping = {"state": "st", "population_m": "pop", "urbanization_pct": "urban_pct"}
    cov = load_state_covariates(external, ["CA", "TX"], column_mapping=mapping)
    assert list(cov[cov["state"] == "CA"]["population_m"])[0] == 39.0
    assert list(cov[cov["state"] == "CA"]["urbanization_pct"])[0] == 95.0
    # unmapped covariate columns default to 0.0
    assert list(cov[cov["state"] == "CA"]["retail_density"])[0] == 0.0


def test_load_state_covariates_requires_source_or_fallback_panel():
    with pytest.raises(ValueError, match="Need either"):
        load_state_covariates(None, ["CA"])
