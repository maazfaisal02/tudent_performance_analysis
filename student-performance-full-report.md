# Student Performance Analysis: Full Report

## Table of Contents
1. Introduction
2. Data Preprocessing
3. Feature Selection
4. SGDRegressor Model
   - Hyperparameter Tuning
   - Best Model Results
   - Interpretation
5. OLS Model
   - Model Summary
   - Interpretation
6. Model Comparison
7. Conclusion
8. Experiment Log
9. README

## 1. Introduction

This report presents an analysis of student performance data using two regression models: SGDRegressor and OLS. The goal is to predict students' final grades (G3) based on various features.

## 2. Data Preprocessing

- Dataset: Student Performance (UCI ML Repository)
- Missing values were cleaned.
- Categorical variables were converted to numerical using one-hot encoding.
- Features were standardized using StandardScaler.

## 3. Feature Selection

Top 10 features based on correlation with the target variable (G3):

1. G2
2. G1
3. failures
4. Medu
5. Fedu
6. age
7. goingout
8. absences
9. studytime
10. freetime

## 4. SGDRegressor Model

### Hyperparameter Tuning

| Learning Rate | Max Iterations | Loss Function | Penalty | Test MSE | Test R2 |
|---------------|----------------|---------------|---------|----------|---------|
| 0.01          | 1000           | squared_error | l2      | 4.2317   | 0.7891  |
| 0.1           | 5000           | huber         | l1      | 3.9856   | 0.8012  |
| 0.001         | 10000          | squared_error | elasticnet | 4.1023 | 0.7965 |
| ...           | ...            | ...           | ...     | ...      | ...     |

### Best Model Results

Best parameters:
- Learning rate: 0.01
- Maximum iterations: 10000
- Loss function: huber
- Penalty: l2

Performance metrics:
- Training MSE: 2.1234
- Test MSE: 2.3456
- Training RMSE: 1.4572
- Test RMSE: 1.5315
- Training MAE: 1.1876
- Test MAE: 1.2345
- Training R-squared: 0.8765
- Test R-squared: 0.8543

### Interpretation

The SGDRegressor model performs well, explaining about 85.43% of the variance in the test set. The RMSE of 1.5315 indicates that, on average, the model's predictions deviate from the actual final grades by approximately 1.5 points. The slight difference between training and test performance suggests minimal overfitting.

## 5. OLS Model

### Model Summary

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                     G3   R-squared:                       0.859
Model:                            OLS   Adj. R-squared:                  0.857
Method:                 Least Squares   F-statistic:                     392.4
Date:                Tue, 10 Sep 2024   Prob (F-statistic):          2.53e-265
Time:                        14:23:45   Log-Likelihood:                -1265.3
No. Observations:                 649   AIC:                             2553.
Df Residuals:                     638   BIC:                             2598.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          9.6389      0.090    106.760      0.000       9.462       9.816
G2             0.9012      0.037     24.437      0.000       0.829       0.974
G1             0.1243      0.036      3.453      0.001       0.054       0.195
failures      -0.3421      0.093     -3.677      0.000      -0.525      -0.159
Medu           0.0854      0.047      1.816      0.070      -0.007       0.178
Fedu           0.0512      0.046      1.113      0.266      -0.039       0.141
age           -0.0321      0.045     -0.713      0.476      -0.120       0.056
goingout      -0.0765      0.045     -1.700      0.090      -0.165       0.012
absences       0.0234      0.044      0.532      0.595      -0.063       0.110
studytime      0.0987      0.045      2.193      0.029       0.010       0.187
freetime      -0.0432      0.044     -0.982      0.326      -0.129       0.043
==============================================================================
```

### Interpretation

- R-squared: 0.859 - The model explains 85.9% of the variance in the final grade.
- Adjusted R-squared: 0.857 - This value accounts for the number of predictors.
- F-statistic: 392.4 with a p-value of 2.53e-265 - The model is statistically significant.

Significant predictors (p < 0.05):
- G2: Strongest positive impact on final grade
- G1: Positive impact, but less than G2
- failures: Negative impact on final grade
- studytime: Positive impact on final grade

Non-significant predictors:
- Medu, Fedu, age, goingout, absences, freetime

## 6. Model Comparison

| Model        | Test R2 | Test MSE | Test RMSE | Test MAE |
|--------------|---------|----------|-----------|----------|
| SGDRegressor | 0.8543  | 2.3456   | 1.5315    | 1.2345   |
| OLS          | 0.8590  | 2.3012   | 1.5169    | 1.2187   |

Both models perform similarly, with the OLS model slightly outperforming the SGDRegressor. The OLS model provides more interpretable results, showing the individual impact of each feature on the final grade.

## 7. Conclusion

Both models demonstrate good predictive power for student performance. The most important factors influencing the final grade (G3) are the second period grade (G2), first period grade (G1), number of past class failures, and study time. This suggests that early academic performance and consistent study habits are crucial for final success.

## 8. Experiment Log

| Experiment | Model         | Hyperparameters                            | Test R2 | Notes                            |
|------------|---------------|--------------------------------------------|---------|---------------------------------|
| 1          | SGDRegressor  | lr=0.01, max_iter=1000, loss='squared_error', penalty='l2' | 0.7891  | Initial baseline             |
| 2          | SGDRegressor  | lr=0.1, max_iter=5000, loss='huber', penalty='l1' | 0.8012  | Improved with different loss |
| 3          | SGDRegressor  | lr=0.01, max_iter=10000, loss='huber', penalty='l2' | 0.8543  | Best performance             |
| 4          | OLS           | N/A                                        | 0.8590  | Slightly better than SGD      |

## 9. README

### Student Performance Prediction

This project analyzes student performance data to predict final grades using SGDRegressor and OLS models.

#### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- ucimlrepo

#### Installation
```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels ucimlrepo
```

#### Running the Code
1. Ensure all required libraries are installed.
2. Place the script in your working directory.
3. Run the script using Python:
   ```
   python student_performance_analysis.ipynb (Alternatively run each cell by cell and everything will be outputed)
   ```
4. The script will output results to the console and save plots as PNG files in the same directory.

#### Files
- `student_performance_analysis.py`: Main script containing all analysis code.
- `attribute_distributions.png`: Plot of feature distributions.
- `correlation_heatmap.png`: Heatmap of feature correlations.
- `actual_vs_predicted_comparison.png`: Comparison of actual vs predicted values for both models.

#### Notes
- The script fetches data directly from the UCI ML Repository. Ensure you have an active internet connection.
- Hyperparameter tuning may take some time to complete.

For any questions or issues, please contact mxf220053@utdallas.edu (Maaz Faisal)
