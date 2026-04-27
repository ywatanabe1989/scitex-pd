"""scitex-pd quickstart: pandas helpers (force_df, to_xyz, melt_cols, mv)."""

import numpy as np
import pandas as pd

import scitex_pd


def main():
    # 1. force_df: coerce a heterogeneous dict / scalar / array into a DataFrame.
    raw = {"alpha": [1, 2, 3], "beta": 7, "gamma": np.array([10, 20, 30])}
    df = scitex_pd.force_df(raw)
    print("force_df ->")
    print(df)
    assert isinstance(df, pd.DataFrame)
    assert {"alpha", "beta", "gamma"}.issubset(df.columns)

    # 2. to_xyz: pivot a wide matrix into long (x, y, z) form,
    # the canonical input to many statistics + heatmap functions.
    mat = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=["row_a", "row_b"],
        columns=["col_x", "col_y", "col_z"],
    )
    long = scitex_pd.to_xyz(mat)
    print("\nto_xyz ->")
    print(long)
    assert len(long) == 6  # 2 rows x 3 cols

    # 3. mv_to_first: reorder columns so a key column appears first.
    df2 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    moved = scitex_pd.mv_to_first(df2, "c")
    print("\nmv_to_first('c') ->", list(moved.columns))
    assert list(moved.columns)[0] == "c"

    # 4. round: round numeric columns to a fixed precision.
    df3 = pd.DataFrame({"x": [1.23456, 2.34567]})
    rounded = scitex_pd.round(df3, 2)
    print("\nround(2) ->")
    print(rounded)
    assert (rounded["x"] - pd.Series([1.23, 2.35])).abs().max() < 1e-9


if __name__ == "__main__":
    main()
