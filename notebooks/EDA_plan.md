
# EDA & Analysis Plan

This document describes the EDA analyses that the project performs (or the student should run on the full dataset).

Minimum 10-15 analyses included:

1. Summary statistics (mean, median, std) for numeric features.
2. Distribution plots (histogram) for age, bmi, and charges.
3. Boxplots to detect outliers for BMI and charges.
4. Scatter plot: BMI vs Charges (to check relationship).
5. Scatter plot: Age vs Charges.
6. Correlation heatmap for numeric features.
7. Grouped aggregation: mean charges by smoker status.
8. Grouped aggregation: mean charges by region.
9. Pairwise relationships using pairplot for selected features.
10. Missing value analysis and counts per column.
11. Categorical value counts for sex, smoker, region.
12. Children count effect: average charges by number of children.

Each of these analyses can be produced by running the notebook `notebooks/EDA_and_training.ipynb` (or by executing the functions provided in `src/train_model.py`).

Example code snippet (pandas):

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/insurance.csv')
print(df.describe())
plt.hist(df['charges'], bins=30)
plt.show()
```
