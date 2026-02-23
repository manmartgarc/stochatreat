import numpy as np
import pandas as pd
import pytest

from stochatreat.treatment import TreatmentAssigner, TreatmentSpec


@pytest.fixture
def spec_equal_probs():
    return TreatmentSpec(treats=2, probs=None, random_state=42)


@pytest.fixture
def prepared_df():
    n = 10
    df = pd.DataFrame(
        {"id": np.arange(n), "stratum_id": [0] * (n // 2) + [1] * (n // 2)}
    )
    return df, "id"


class TestTreatmentSpec:
    def test_equal_probs(self):
        spec = TreatmentSpec(treats=2, probs=None, random_state=0)
        assert spec.treatment_ids == [0, 1]
        np.testing.assert_array_almost_equal(spec.probs, [0.5, 0.5])

    def test_custom_probs(self):
        spec = TreatmentSpec(treats=2, probs=[0.25, 0.75], random_state=0)
        np.testing.assert_array_almost_equal(spec.probs, [0.25, 0.75])

    def test_probs_not_sum_to_one(self):
        with pytest.raises(ValueError, match="probabilities must add up to 1"):
            TreatmentSpec(treats=2, probs=[0.1, 0.2], random_state=0)

    def test_probs_length_mismatch(self):
        with pytest.raises(ValueError, match="number of probabilities"):
            TreatmentSpec(treats=3, probs=[0.5, 0.5], random_state=0)

    def test_treat_mask_length(self, spec_equal_probs):
        assert (
            len(spec_equal_probs.treat_mask)
            == spec_equal_probs.lcm_prob_denominators
        )

    def test_treat_mask_contents(self, spec_equal_probs):
        assert set(spec_equal_probs.treat_mask) == {0, 1}

    def test_lcm_two_equal(self, spec_equal_probs):
        assert spec_equal_probs.lcm_prob_denominators == 2  # noqa: PLR2004

    def test_lcm_thirds(self):
        spec = TreatmentSpec(treats=2, probs=[1 / 3, 2 / 3], random_state=0)
        assert spec.lcm_prob_denominators == 3  # noqa: PLR2004

    def test_rng_reproducible(self):
        spec1 = TreatmentSpec(treats=2, probs=None, random_state=42)
        spec2 = TreatmentSpec(treats=2, probs=None, random_state=42)
        assert spec1.rng.rand() == spec2.rng.rand()


class TestTreatmentAssigner:
    def test_output_columns(self, spec_equal_probs, prepared_df):
        df, idx_col = prepared_df
        result = TreatmentAssigner(spec_equal_probs).assign(df, idx_col)
        assert set(result.columns) == {"id", "stratum_id", "treat"}

    def test_no_missing_ids(self, spec_equal_probs, prepared_df):
        df, idx_col = prepared_df
        result = TreatmentAssigner(spec_equal_probs).assign(df, idx_col)
        assert set(result["id"]) == set(df["id"])

    def test_treat_dtype(self, spec_equal_probs, prepared_df):
        df, idx_col = prepared_df
        result = TreatmentAssigner(spec_equal_probs).assign(df, idx_col)
        assert result["treat"].dtype == "Int64"

    def test_no_null_treats(self, spec_equal_probs, prepared_df):
        df, idx_col = prepared_df
        result = TreatmentAssigner(spec_equal_probs).assign(df, idx_col)
        assert result["treat"].isnull().sum() == 0

    def test_treats_in_valid_range(self):
        spec = TreatmentSpec(treats=3, probs=None, random_state=42)
        n = 12
        df = pd.DataFrame(
            {"id": np.arange(n), "stratum_id": [0] * (n // 2) + [1] * (n // 2)}
        )
        result = TreatmentAssigner(spec).assign(df, "id")
        assert set(result["treat"]).issubset({0, 1, 2})

    def test_reproducible(self, prepared_df):
        df, idx_col = prepared_df
        spec1 = TreatmentSpec(treats=2, probs=None, random_state=7)
        spec2 = TreatmentSpec(treats=2, probs=None, random_state=7)
        r1 = TreatmentAssigner(spec1).assign(df.copy(), idx_col)
        r2 = TreatmentAssigner(spec2).assign(df.copy(), idx_col)
        pd.testing.assert_series_equal(
            r1["treat"].reset_index(drop=True),
            r2["treat"].reset_index(drop=True),
        )
