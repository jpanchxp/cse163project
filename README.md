# CSE 163 Final Project

This project analyzes the OECD PIAAC Cycle 2 dataset to examine relationships between literacy, reading behaviors, income, education, and occupational skill.

## Setup

Python 3 is required. Install the required libraries:

    pip install pandas numpy matplotlib seaborn scipy scikit-learn

Download the PIAAC Cycle 2 U.S. dataset and place "prgusap2.csv" in the same directory as the Python files.

## Files

- `data_processing.py`: Loads and cleans the PIAAC dataset and calculates
  literacy scores.
- `analysis.py`: Runs the statistical analyses and machine learning models and
  generates the figures used in the report.
- `project_test.py`: Tests data cleaning and preparation for the analyses.
- `test_lit_data.csv`: Small dataset used by `project_test.py`.
- `report.pdf`: Final project report.

## Running the Project

1. Make sure "prgusap2.csv" is in the project directory.
2. From the project directory, run:
    
    python analysis.py

    -> This reproduces the statistical results and figures used in the report

3. To run the tests:

    python project_test.py

    -> Successful tests will print "All tests passed!"

No file paths need to be changed if all files are kept in the same directory.
