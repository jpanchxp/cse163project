"""
CSE 163 - EDA
Jyothi Panchapagesan
TAs: Sheamin Kim and Katie Gower

This file loads and cleans the PIAAC dataset for analysis.
"""

import numpy as np
import pandas as pd

DATA = "prgusap2.csv"


def load_and_clean_data(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and cleans the PIAAC dataset and computes literacy scores.

    filename is the path to the PIAAC CSV file.

    Returns the cleaned analysis DataFrame and the original DataFrame.
    """
    # Load dataset
    raw_litdata = pd.read_csv(filename, sep=";", low_memory=False)

    # Select relevant variables
    selected_cols = [
            "PVLIT1",
            "PVLIT2",
            "PVLIT3",
            "PVLIT4",
            "PVLIT5",
            "PVLIT6",
            "PVLIT7",
            "PVLIT8",
            "PVLIT9",
            "PVLIT10",
            "MONTHLYINCPR",
            "READHOMEC2_T1",
            "READWORKC2_T1",
            "EDCAT6_TC1",
            "GENDER_R",
            "PAIDWORK12",
            "ISCOSKIL4"]

    analysis_df = raw_litdata[selected_cols].copy()

    analysis_df = analysis_df.replace({
        ".": np.nan,
        ".n": np.nan,
        ".v": np.nan,
        ".a": np.nan,
        ".u": np.nan
    })

    numeric_cols = [
        "MONTHLYINCPR",
        "READHOMEC2_T1",
        "READWORKC2_T1",
        "EDCAT6_TC1",
        "PAIDWORK12",
        "ISCOSKIL4"
    ]

    analysis_df[numeric_cols] = analysis_df[numeric_cols].apply(
        pd.to_numeric
    )

    analysis_df["literacy_score"] = analysis_df[[
            "PVLIT1",
            "PVLIT2",
            "PVLIT3",
            "PVLIT4",
            "PVLIT5",
            "PVLIT6",
            "PVLIT7",
            "PVLIT8",
            "PVLIT9",
            "PVLIT10"]].mean(axis=1)

    return analysis_df, raw_litdata
