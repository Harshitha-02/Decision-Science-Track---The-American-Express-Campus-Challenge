# Predicting Winner in a T-20 Game | AMEX Campus Super Bowl Challenge

## Project Overview

This project aims to predict the winner of T-20 cricket matches using an ensemble model. The model integrates Catboost, XGBoost, GBM, and LightGBM, achieving a prediction accuracy of 71.84%. The project involves extensive feature engineering and hyper-parameter tuning to enhance the model's performance.

## Key Features

- **Ensemble Model:** Utilizes Catboost, XGBoost, GBM, and LightGBM to predict the match winner.
- **High Accuracy:** Achieved an accuracy rate of 71.84%.
- **Feature Engineering:** Engineered over 20 features from Batsmen, Bowlers, and Matches datasets to improve model precision.
- **Hyper-Parameter Tuning:** Applied advanced techniques such as Optuna and Hyperopt, leading to a 13.22% increase in accuracy.

## Data Sources

- **Batsmen Dataset:** Contains performance metrics for individual batsmen.
- **Bowlers Dataset:** Contains performance metrics for individual bowlers.
- **Matches Dataset:** Contains historical data of T-20 matches, including outcomes and relevant statistics.

## Features Engineered

The following features were engineered to enhance the model's predictive power:

- **Team Performance:** Metrics such as team win percentage in the last 5 matches, average score on specific grounds, and team experience score.
- **Player Performance:** Individual performance ratios for batsmen and bowlers, including average runs per wicket and wickets conceded.
- **Match Context:** Factors like the toss decision, ground conditions, and night game performance.

## Hyper-Parameter Tuning

- **Optuna:** Used for automated hyper-parameter optimization, exploring various configurations to find the best-performing model parameters.
- **Hyperopt:** Applied to further refine the model by conducting an extensive search over the hyper-parameter space, resulting in significant performance gains.

## Results

The ensemble model achieved a notable accuracy of 71.84%, a significant improvement of 13.22% from the baseline model. The advanced feature engineering and hyper-parameter tuning techniques played a crucial role in enhancing the model's predictive accuracy.

## Conclusion

This project showcases the power of ensemble models and advanced hyper-parameter tuning techniques in predicting the outcomes of T-20 cricket matches. The feature engineering process was critical in extracting valuable insights from the data, leading to a high-performing predictive model.
