"""
CSE 163 - Analysis
Jyothi Panchapagesan
TAs: Sheamin Kim and Katie Gower

This program analyzes PIAAC literacy data to investigate relationships
among literacy, reading behavior, education, income, and occupational skill.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_processing import load_and_clean_data
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay

DATA = "prgusap2.csv"


def print_summaries(analysis_df: pd.DataFrame,
                    raw_litdata: pd.DataFrame) -> None:
    """
    Prints dataset dimensions, missing-value information, descriptive
    statistics, and category counts for the PIAAC analysis.

    analysis_df is the cleaned PIAAC dataset used for analysis
    raw_litdata is the original PIAAC dataset
    """
    print("Original Dataset shape:")
    print(raw_litdata.shape)

    print("Analysis Dataset shape:")
    print(analysis_df.shape)

    print("\nMissing-value counts:")
    print(analysis_df.isna().sum())

    print("\nMissing-value percentages:")
    print(analysis_df.isna().sum() / len(analysis_df) * 100)

    print("\nLiteracy score summary:")
    print(analysis_df["literacy_score"].describe())

    print("\nReading-at-home summary:")
    print(analysis_df["READHOMEC2_T1"].describe())

    print("\nReading-at-work summary:")
    print(analysis_df["READWORKC2_T1"].describe())

    print("\nMonthly income categories:")
    print(analysis_df["MONTHLYINCPR"].value_counts())

    print("\nEducation categories:")
    print(analysis_df["EDCAT6_TC1"].value_counts())

    print("\nGender categories:")
    print(analysis_df["GENDER_R"].value_counts())


def create_visualizations(analysis_df: pd.DataFrame) -> None:
    """
    Creates and saves visualizations summarizing literacy scores and their
    relationships with reading behavior, income, education, and gender.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    sns.histplot(data=analysis_df, x="literacy_score")
    plt.title("Distribution of PIAAC Literacy Scores")
    plt.xlabel("Literacy Score")
    plt.ylabel("Individuals Count")
    plt.savefig("literacy_distribution.png")
    plt.show()

    sns.regplot(
        data=analysis_df,
        x="READHOMEC2_T1",
        y="literacy_score",
        scatter_kws={"color": "blue", "alpha": 0.3},
        line_kws={"color": "red"}
    )
    plt.title("Literacy vs Reading Skills Used at Home")
    plt.xlabel("Reading at Home Index")
    plt.ylabel("Literacy Score")
    plt.savefig("literacy_home_reading.png")
    plt.show()

    sns.regplot(
        data=analysis_df,
        x="READWORKC2_T1",
        y="literacy_score",
        scatter_kws={"color": "blue", "alpha": 0.3},
        line_kws={"color": "red"}
    )
    plt.title("Literacy vs Reading Skills Used at Work")
    plt.xlabel("Reading at Work Index")
    plt.ylabel("Literacy Score")
    plt.savefig("literacy_work_reading.png")
    plt.show()

    sns.boxplot(
        data=analysis_df,
        x="MONTHLYINCPR",
        y="literacy_score"
    )
    plt.title("Literacy Score by Monthly Income Percentile Category")
    plt.xlabel("Monthly Income Percentile Category")
    plt.ylabel("Literacy Score")
    plt.savefig("literacy_income.png")
    plt.show()

    sns.boxplot(
        data=analysis_df,
        x="EDCAT6_TC1",
        y="literacy_score"
    )
    plt.title("Literacy Score by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("Literacy Score")
    plt.savefig("literacy_education.png")
    plt.show()

    sns.boxplot(
        data=analysis_df,
        x="GENDER_R",
        y="literacy_score"
    )
    plt.title("Literacy Score by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Literacy Score")
    plt.xticks([0, 1], ["Male", "Female"])
    plt.savefig("literacy_gender.png")
    plt.show()


def get_rq1_data(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns complete rows needed for the literacy-income analysis.
    """
    return analysis_df[
        ["literacy_score", "MONTHLYINCPR", "EDCAT6_TC1"]
    ].dropna()


def analyze_literacy_income(analysis_df: pd.DataFrame) -> None:
    """
    Reports the associations among literacy score, income category,
    and education level.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    rq1_df = get_rq1_data(analysis_df)

    correlation, p_value = spearmanr(
        rq1_df["literacy_score"],
        rq1_df["MONTHLYINCPR"]
    )

    correlation2, p_value2 = spearmanr(
        rq1_df["EDCAT6_TC1"],
        rq1_df["MONTHLYINCPR"]
    )

    correlation3, p_value3 = spearmanr(
        rq1_df["EDCAT6_TC1"],
        rq1_df["literacy_score"]
    )

    print()
    print("Analysis of Literacy Score - Monthly Income Group Correlation")
    print(f"Spearman correlation: {correlation}")
    print(f"p_value: {p_value}")
    print()

    print("Analysis of Education - Monthly Income Group Correlation")
    print(f"Spearman correlation: {correlation2}")
    print(f"p_value: {p_value2}")
    print()

    print("Analysis of Education - Literacy Score Correlation")
    print(f"Spearman correlation: {correlation3}")
    print(f"p_value: {p_value3}")
    print()


def analyze_literacy_income_by_edc(analysis_df: pd.DataFrame) -> None:
    """
    Reports and visualizes the association between literacy score
    and income category within each education level.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    rq1_df = get_rq1_data(analysis_df)
    eds = [1, 2, 3, 4, 5, 6]
    corrs = []

    for ed_level in eds:
        education_data = rq1_df[
            rq1_df["EDCAT6_TC1"] == ed_level
        ]
        correlation, p_value = spearmanr(
            education_data["literacy_score"],
            education_data["MONTHLYINCPR"]
        )
        corrs.append(correlation)
        print(f"Education category: {ed_level}")
        print(f"Number of observations: {len(education_data)}")
        print(f"Spearman correlation: {correlation}")
        print(f"p_value: {p_value}")
        print()

    corr_df = pd.DataFrame({
        "Education Level": eds,
        "Spearman Correlation": corrs
    })
    sns.barplot(data=corr_df, x="Education Level", y="Spearman Correlation")
    plt.axhline(0)
    plt.xlabel("Education Level")
    plt.ylabel("Spearman Correlation (ρ)")
    plt.title("Literacy-Income Association Within Education Levels")
    plt.savefig("spearman_corr_edlitincome.png")
    plt.show()


def get_rq3_data(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns complete rows needed for income prediction.
    """
    return analysis_df[
        [
            "MONTHLYINCPR",
            "literacy_score",
            "READHOMEC2_T1",
            "READWORKC2_T1",
            "EDCAT6_TC1",
            "GENDER_R"
        ]
    ].dropna()


def analyze_reading_behaviors(analysis_df: pd.DataFrame) -> None:
    """
    Reports and visualizes associations between reading behavior, literacy
    score, and monthly income category.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    home_lit = analysis_df[
        ["READHOMEC2_T1", "literacy_score"]
    ].dropna()

    work_lit = analysis_df[
        ["READWORKC2_T1", "literacy_score"]
    ].dropna()

    home_income = analysis_df[
        ["READHOMEC2_T1", "MONTHLYINCPR"]
    ].dropna()

    work_income = analysis_df[
        ["READWORKC2_T1", "MONTHLYINCPR"]
    ].dropna()

    correlation, p_value = spearmanr(
        home_lit["READHOMEC2_T1"],
        home_lit["literacy_score"]
    )

    correlation2, p_value2 = spearmanr(
        work_lit["READWORKC2_T1"],
        work_lit["literacy_score"]
    )

    correlation3, p_value3 = spearmanr(
        home_income["READHOMEC2_T1"],
        home_income["MONTHLYINCPR"]
    )

    correlation4, p_value4 = spearmanr(
        work_income["READWORKC2_T1"],
        work_income["MONTHLYINCPR"]
    )

    print("Reading at Home - Literacy")
    print(f"Spearman correlation: {correlation}")
    print(f"p_value: {p_value}")

    print("Reading at Work - Literacy")
    print(f"Spearman correlation: {correlation2}")
    print(f"p_value: {p_value2}")

    print("Reading at Home - Income")
    print(f"Spearman correlation: {correlation3}")
    print(f"p_value: {p_value3}")

    print("Reading at Work - Income")
    print(f"Spearman correlation: {correlation4}")
    print(f"p_value: {p_value4}")

    corr_df = pd.DataFrame({
        "Outcome": [
            "Literacy",
            "Literacy",
            "Income",
            "Income"
        ],
        "Reading Location": [
            "Home",
            "Work",
            "Home",
            "Work"
        ],
        "Spearman Correlation": [
            correlation,
            correlation2,
            correlation3,
            correlation4
        ]
    })

    sns.barplot(
        data=corr_df,
        x="Outcome",
        y="Spearman Correlation",
        hue="Reading Location"
    )

    plt.axhline(y=0)
    plt.xlabel("")
    plt.ylabel("Spearman Correlation (ρ)")
    plt.title("Reading Behavior Associations with Literacy and Income")
    plt.savefig("spearman_corr_workvhome.png")
    plt.show()


def analyze_income_prediction(analysis_df: pd.DataFrame) -> None:
    """
    Compares machine-learning models for predicting monthly income
    category from literacy, reading behavior, education, and gender.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    rq3_df = get_rq3_data(analysis_df)

    y = rq3_df["MONTHLYINCPR"]
    x = rq3_df[
        [
            "literacy_score",
            "READHOMEC2_T1",
            "READWORKC2_T1",
            "EDCAT6_TC1",
            "GENDER_R",
        ]
    ]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Decision Tree Accuracy: {accuracy}")

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(Y_test, rf_predictions)
    print(f"Random Forest Accuracy: {rf_accuracy}")

    baseline_accuracy = Y_test.value_counts(normalize=True).max()
    print(f"Baseline Accuracy: {baseline_accuracy}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, Y_train)
    knn_predictions = knn.predict(X_test_scaled)

    knn_accuracy = accuracy_score(Y_test, knn_predictions)
    print(f"KNN Accuracy: {knn_accuracy}")

    dt_f1 = f1_score(Y_test, predictions, average="macro")
    rf_f1 = f1_score(Y_test, rf_predictions, average="macro")
    knn_f1 = f1_score(Y_test, knn_predictions, average="macro")

    print(f"Decision Tree Macro F1: {dt_f1}")
    print(f"Random Forest Macro F1: {rf_f1}")
    print(f"KNN Macro F1: {knn_f1}")

    ConfusionMatrixDisplay.from_predictions(
        Y_test,
        rf_predictions
    )

    plt.title("Random Forest Income Category Predictions")
    plt.xlabel("Predicted Income Category")
    plt.ylabel("Actual Income Category")
    plt.savefig("random_forest_confusion_matrix.png")
    plt.show()

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": [
            "Literacy Score",
            "Reading at Home",
            "Reading at Work",
            "Education Level",
            "Gender"
        ],
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    sns.barplot(
        data=importance_df,
        y="Importance",
        x="Feature"
    )

    plt.title("Random Forest Feature Importance")
    plt.xlabel("")
    plt.ylabel("Feature Importance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("random_forest_feature_importance.png")
    plt.show()


def analyze_occupation_literacy(analysis_df: pd.DataFrame) -> None:
    """
    Reports and visualizes the association between
    occupational skill level and literacy score.

    analysis_df is the cleaned PIAAC dataset used for analysis.
    """
    occ_skill_lit = analysis_df[
        ["ISCOSKIL4", "literacy_score"]
    ].dropna()

    correlation, p_value = spearmanr(
        occ_skill_lit["ISCOSKIL4"],
        occ_skill_lit["literacy_score"]
    )

    print("Occupational Skill Level - Literacy")
    print(f"Spearman correlation: {correlation}")
    print(f"p_value: {p_value}")

    sns.boxplot(data=occ_skill_lit, x="ISCOSKIL4", y="literacy_score")
    plt.xlabel("Occupational Skill Level")
    plt.ylabel("Literacy Score")
    plt.title("Occupational Skill Level Association with Literacy Score")
    plt.xticks(
        [0, 1, 2, 3],
        [
            "Skilled",
            "Semi-skilled\nWhite-collar",
            "Semi-skilled\nBlue-collar",
            "Elementary"
        ]
    )
    plt.tight_layout()
    plt.savefig("occ_skill_level_lit_score.png")
    plt.show()


def main():
    """
    Runs the complete PIAAC analysis workflow.
    """
    analysis_df, raw_litdata = load_and_clean_data(DATA)
    print_summaries(analysis_df, raw_litdata)
    create_visualizations(analysis_df)
    analyze_literacy_income(analysis_df)
    analyze_literacy_income_by_edc(analysis_df)
    analyze_reading_behaviors(analysis_df)
    analyze_income_prediction(analysis_df)
    analyze_occupation_literacy(analysis_df)


if __name__ == "__main__":
    main()
