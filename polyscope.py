#!/usr/bin/env python3
"""
PolyScope: Streamlit app to explore saved manager archives from LM-Polygraph.
"""
import io

import streamlit as st
st.set_page_config(page_title="PolyScope")
import torch
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import json
import plotly.express as px


def load_archive(uploaded_file):
    """Load the torch-saved archive and return meta dict and DataFrame."""
    file_bytes = uploaded_file.read()
    raw = torch.load(
        io.BytesIO(file_bytes), map_location="cpu", weights_only=False
    )
    meta = {k: v for k, v in raw.items() if k not in ("estimations", "stats", "gen_metrics")}

    N = None
    if "stats" in raw and raw["stats"]:
        first = next(iter(raw["stats"].values()))
        N = len(first)
    elif "estimations" in raw and raw["estimations"]:
        first = next(iter(raw["estimations"].values()))
        N = len(first)
    elif "gen_metrics" in raw and raw["gen_metrics"]:
        first = next(iter(raw["gen_metrics"].values()))
        N = len(first)
    else:
        st.error("Archive contains no records in stats/estimations/gen_metrics.")
        return meta, None

    df = pd.DataFrame(index=range(N))

    est_cols = []
    for (_seq, method), vals in raw.get("estimations", {}).items():
        df[method] = vals
        est_cols.append(method)

    for stat_name, vals in raw.get("stats", {}).items():
        df[stat_name] = vals

    met_cols = []
    for (_seq, metric), vals in raw.get("gen_metrics", {}).items():
        df[metric] = vals
        met_cols.append(metric)

    return meta, df, est_cols, met_cols


def identify_columns(df, est_cols, met_cols):
    """Identify stat and numeric stat columns in the DataFrame."""
    stat_cols = [c for c in df.columns if c not in est_cols + met_cols]
    numeric_stats = [c for c in stat_cols if is_numeric_dtype(df[c])]
    return stat_cols, numeric_stats


def compute_quantiles(df, numeric_stats):
    """
    Compute quantiles at 5% increments for numeric columns.
    Returns a dict mapping column name to a Series of quantile values indexed by fraction.
    """
    quantiles = {}
    # Quantile fractions from 0.0 to 1.0 in steps of 0.05
    q_points = np.arange(0.0, 1.0001, 0.05)
    for col in numeric_stats:
        quantiles[col] = df[col].quantile(q_points)
    return quantiles


def filter_dataframe(df, id_range, feature_bounds, quantiles):
    """Filter the DataFrame by id_range and quantile bounds per feature."""
    mask = (df.index >= id_range[0]) & (df.index <= id_range[1])
    for feature, (low_q, high_q) in feature_bounds.items():
        low_val = quantiles[feature].loc[low_q]
        high_val = quantiles[feature].loc[high_q]
        mask &= (df[feature] >= low_val) & (df[feature] <= high_val)
    return df.loc[mask]


def compute_rejection_curve(df, uncertainty_col, metric_col, steps=100, ascending=False):
    """
    Compute rejection curve: fraction rejected vs average metric of remaining.
    If ascending is False, removes highest uncertainty first (descending); if True, removes lowest metric first (oracle).
    """
    sorted_df = df.sort_values(by=uncertainty_col, ascending=ascending)
    N = len(sorted_df)
    fractions = np.linspace(0.0, 1.0, num=steps)
    records = []
    for frac in fractions:
        k = int(frac * N)
        remaining = sorted_df.iloc[k:]
        metric_val = remaining[metric_col].mean() if len(remaining) > 0 else np.nan
        records.append({"fraction_rejected": frac, "metric": metric_val})
    return pd.DataFrame(records)


def main():
    st.title("PolyScope")

    uploaded = st.sidebar.file_uploader(
        "Upload archive (.pt, .pth, .man)", type=["pt", "pth", "man"]
    )
    if not uploaded:
        st.info(
            "Please upload a .pt, .pth, or .man file containing benchmark results."
        )
        return

    meta, df, est_cols, met_cols = load_archive(uploaded)
    if df is None:
        return

    try:
        with open("column_acronyms.json", "r") as f:
            acronyms = json.load(f)
    except Exception:
        acronyms = {}

    df.rename(columns=acronyms, inplace=True)
    est_cols = [acronyms.get(c, c) for c in est_cols]
    met_cols = [acronyms.get(c, c) for c in met_cols]

    stat_cols, numeric_stats = identify_columns(df, est_cols, met_cols)

    # Precompute quantiles for all numeric columns
    numeric_cols_all = df.select_dtypes(include="number").columns.tolist()
    quantiles = compute_quantiles(df, numeric_cols_all) if numeric_cols_all else {}

    st.sidebar.header("Meta Info")

    st.sidebar.header("Filters")
    id_min, id_max = int(df.index.min()), int(df.index.max())
    id_range = st.sidebar.slider("ID range", id_min, id_max, (id_min, id_max))

    # Column selection (default to input_texts if present)
    cols = list(df.columns)
    default_cols = ["input_texts"] if "input_texts" in cols else cols
    display_cols = st.sidebar.multiselect(
        "Columns to display", cols, default=default_cols
    )

    # Quantile filters for displayed numeric columns
    feature_bounds = {}
    for col in display_cols:
        if col in quantiles:
            q_opts = list(quantiles[col].index)
            low, high = st.sidebar.select_slider(
                f"{col} quantile range", options=q_opts, value=(0.0, 1.0)
            )
            feature_bounds[col] = (low, high)

    # Apply filters (ID range + quantile bounds)
    filtered = filter_dataframe(df, id_range, feature_bounds, quantiles)

    st.subheader("Filtered Data")
    st.dataframe(filtered[display_cols])

    st.sidebar.header("Rejection Curves")
    if est_cols and met_cols:
        u_choices = st.sidebar.multiselect(
            "Uncertainty methods", est_cols, default=est_cols[:1]
        )
        m_choice = st.sidebar.selectbox("Quality metric", met_cols)
        if u_choices and m_choice:
            # Oracle curve based on true metric ranking (remove lowest metric first)
            rej_oracle = compute_rejection_curve(
                filtered, m_choice, m_choice, ascending=True
            )
            combined = pd.DataFrame({
                "fraction_rejected": rej_oracle["fraction_rejected"],
                "oracle": rej_oracle["metric"],
            })
            for u in u_choices:
                rej = compute_rejection_curve(filtered, u, m_choice, ascending=False)
                combined[u] = rej["metric"].values

            all_cols = u_choices + ["oracle"]
            min_val = combined[all_cols].min().min()
            max_val = combined[all_cols].max().max()
            y0 = min_val - 0.02 * (max_val - min_val)

            fig = px.line(
                combined,
                x="fraction_rejected",
                y=all_cols,
                labels={
                    "value": m_choice,
                    "fraction_rejected": "Fraction Rejected",
                    "variable": "Curve",
                },
            )
            fig.update_yaxes(range=[y0, max_val])
            st.subheader("Rejection Curves")
            st.plotly_chart(fig, use_container_width=True)
            metrics_dict = meta.get("metrics", {})
            if metrics_dict:
                inv_acro = {v: k for k, v in acronyms.items()}
                raw_methods = [inv_acro.get(m, m) for m in est_cols]
                raw_metrics = [inv_acro.get(m, m) for m in met_cols]
                prr_rows = []
                for disp_m, raw_m in zip(est_cols, raw_methods):
                    row = []
                    for disp_n, raw_n in zip(met_cols, raw_metrics):
                        key = ("sequence", raw_m, raw_n, "prr_0.5_normalized")
                        row.append(metrics_dict.get(key, np.nan))
                    prr_rows.append(row)
                prr_df = pd.DataFrame(prr_rows, index=est_cols, columns=met_cols)
                st.subheader("PRR Scores")
                st.dataframe(prr_df)


if __name__ == "__main__":
    main()
