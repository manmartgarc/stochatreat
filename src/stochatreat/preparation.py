"""Input validation and data preparation for treatment assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

MIN_ROW_N = 2


class DataPreparator:
    """Validates and prepares input data for treatment assignment.

    Performs all input validation and transforms the raw DataFrame into
    a clean form with stratum IDs assigned, sorted by idx_col, and
    optionally subsampled.

    Args:
        data: Input DataFrame.
        stratum_cols: Column(s) to stratify over.
        idx_col: Column with unique unit identifiers. If None, the
            DataFrame index is used.
        size: Desired sample size. If None, the full dataset is used.
        random_state: Seed for reproducible sampling.

    Raises:
        ValueError: If the DataFrame is empty, has fewer than 2 rows,
            if idx_col values are not unique, or if size exceeds the
            number of rows.
        TypeError: If idx_col is not a string or None.
        KeyError: If any column in stratum_cols or idx_col is missing
            from data.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        stratum_cols: list[str] | str,
        idx_col: str | None,
        size: int | None,
        random_state: int | None,
    ) -> None:
        """Store parameters for later use in prepare()."""
        self._data = data.copy()
        self._stratum_cols = stratum_cols
        self._idx_col = idx_col
        self._size = size
        self._random_state = random_state

    def prepare(self) -> tuple[pd.DataFrame, str]:
        """Validate inputs and return a prepared DataFrame with stratum IDs.

        Returns:
            A tuple of (prepared_df, resolved_idx_col) where
            resolved_idx_col is the name of the id column (which may
            have been synthesised from the index).

        Raises:
            ValueError: See class docstring.
            TypeError: See class docstring.
            KeyError: See class docstring.

        """
        data = self._data

        if data.empty:
            msg = "Make sure that your dataframe is not empty."
            raise ValueError(msg)
        if len(data) < MIN_ROW_N:
            msg = "Your dataframe at least needs to have 2 rows."
            raise ValueError(msg)

        idx_col = self._idx_col
        if idx_col is None:
            data = data.rename_axis("index", axis="index").reset_index()
            idx_col = "index"
        elif not isinstance(idx_col, str):
            msg = "idx_col has to be a string."
            raise TypeError(msg)

        stratum_cols = (
            [self._stratum_cols]
            if isinstance(self._stratum_cols, str)
            else list(self._stratum_cols)
        )
        missing = [
            c for c in [*stratum_cols, idx_col] if c not in data.columns
        ]
        if missing:
            msg = f"Columns not found in data: {missing}"
            raise KeyError(msg)

        if data[idx_col].duplicated(keep=False).sum() > 0:
            msg = "The values in idx_col are not unique."
            raise ValueError(msg)

        if self._size is not None and self._size > len(data):
            msg = "Size argument is larger than the sample universe."
            raise ValueError(msg)

        data = data.sort_values(by=idx_col)
        data["stratum_id"] = data.groupby(
            stratum_cols, observed=False
        ).ngroup()
        data = data[[idx_col, "stratum_id"]].copy()

        if self._size is not None:
            size = int(self._size)
            strata_fracs = (
                data["stratum_id"].value_counts(normalize=True).sort_index()
            )
            reduced_sizes = (strata_fracs * size).round().astype(int)
            data = data.groupby("stratum_id").apply(
                lambda x: x.sample(
                    n=reduced_sizes[x.name],
                    random_state=self._random_state,
                ),
                include_groups=False,
            )
            data["stratum_id"] = data.index.get_level_values(0)
            data = data.droplevel(level="stratum_id")

        return data, idx_col
