"""
CSE 163 - Final Project Testing
Jyothi Panchapagesan
TAs: Sheamin Kim and Katie Gower

Tests the data cleaning and analysis preparation used in the PIAAC project.
"""

import numpy as np
from data_processing import load_and_clean_data
from analysis import get_rq1_data, get_rq3_data

TEST_DATA = "test_lit_data.csv"


def test_load_and_clean_data():
    """
    Tests that the dataset loads correctly and contains the expected
    shape and computed literacy_score column.
    """
    result, raw_data = load_and_clean_data(TEST_DATA)

    assert result.shape == (5, 18)
    assert "literacy_score" in result.columns


def test_lit_score():
    """
    Tests that literacy scores are computed correctly as the average
    of ten literacy variables.
    """
    result, raw_data = load_and_clean_data(TEST_DATA)

    assert result.loc[0, "literacy_score"] == 145
    assert result.loc[1, "literacy_score"] == 245
    assert result.loc[2, "literacy_score"] == 345
    assert result.loc[3, "literacy_score"] == 195
    assert result.loc[4, "literacy_score"] == 295


def test_missing_values():
    """
    Tests that special missing-value codes are converted to NaN
    """
    result, raw_data = load_and_clean_data(TEST_DATA)

    assert np.isnan(result.loc[2, "MONTHLYINCPR"])
    assert np.isnan(result.loc[2, "READHOMEC2_T1"])
    assert np.isnan(result.loc[2, "READWORKC2_T1"])
    assert np.isnan(result.loc[2, "ISCOSKIL4"])

    assert np.isnan(result.loc[3, "READWORKC2_T1"])
    assert np.isnan(result.loc[3, "EDCAT6_TC1"])
    assert np.isnan(result.loc[3, "PAIDWORK12"])


def test_valid_values():
    """
    Tests that valid non-missing values remain unchanged after
    data cleaning
    """
    result, raw_data = load_and_clean_data(TEST_DATA)

    assert result.loc[0, "MONTHLYINCPR"] == 1
    assert result.loc[0, "READHOMEC2_T1"] == 2.5
    assert result.loc[0, "READWORKC2_T1"] == 3.0
    assert result.loc[0, "GENDER_R"] == 1
    assert result.loc[1, "GENDER_R"] == 2


def test_get_rq1_data():
    """
    Tests that RQ1 data contains only complete required observations.
    """
    analysis_df, raw_data = load_and_clean_data(TEST_DATA)

    result = get_rq1_data(analysis_df)

    assert result.shape == (3, 3)
    assert list(result.columns) == [
        "literacy_score",
        "MONTHLYINCPR",
        "EDCAT6_TC1"
    ]


def test_get_rq3_data():
    """
    Tests that RQ3 data contains the expected complete modeling variables.
    """
    analysis_df, raw_data = load_and_clean_data(TEST_DATA)

    result = get_rq3_data(analysis_df)

    assert result.shape == (3, 6)
    assert list(result.columns) == [
        "MONTHLYINCPR",
        "literacy_score",
        "READHOMEC2_T1",
        "READWORKC2_T1",
        "EDCAT6_TC1",
        "GENDER_R"
    ]


def main():
    """
    Runs all test functions.
    """
    test_load_and_clean_data()
    test_lit_score()
    test_missing_values()
    test_valid_values()
    test_get_rq1_data()
    test_get_rq3_data()
    print("All tests passed!")


if __name__ == "__main__":
    main()
