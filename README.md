# Bus-Ticket-Demand-Prediction
Predict bus seat demand 15 days ahead using LightGBM with feature engineering and Optuna tuning.

# Bus Seat Demand Forecasting – Analytics Vidhya Hackathon

This project predicts the number of seats required (`final_seatcount`) for intercity bus routes using data provided in the Analytics Vidhya hackathon.

##  Dataset Overview

The dataset contains the following key tables:
- **Train**: Contains historical booking information.
- **Test**: Used for prediction submission.
- **Searches**: Number of searches per route.
- **Bookings**: Number of bookings per route.
- **Route Details**: Metadata about the route.

---

##  Feature Engineering

Performed after merging all data sources:
- `search_to_book_ratio`: Ratio of searches to bookings.
- `weekday_num`: Extracted from `doj` (day of journey).
- `is_intra_state`: Indicates if src and dest are in the same state.
- `route_freq`: Frequency of each route.
- Grouped statistics:
  - `avg_seatcount_per_route`
  - `median_seatcount_per_route`
  - `count_of_trips_per_route`
- Interaction:
  - `bookings_x_weekday`: Interaction of bookings and weekday.

---

##  Model Training

###  Preprocessing:
- Categorical columns were cast as `'category'` dtype.
- Log transformation was applied to the target `final_seatcount` using `log1p()` for normalizing skew.

###  Model Used:
- **LightGBM Regressor** (`LGBMRegressor`)
  - Gradient Boosting-based tree ensemble.
  - Suitable for handling categorical features natively.

###  Hyperparameter Tuning:
- Used **Optuna** to optimize:
  - `learning_rate`, `num_leaves`, `max_depth`, `feature_fraction`, etc.
- Evaluation Metric: **RMSE** on reverse-transformed predictions.

###  Cross Validation:
- 5-Fold cross-validation using `KFold`.
- RMSE was calculated on the original (expm1-transformed) scale.

---

## 📈 Feature Importance

Plotted LightGBM’s feature importances. Final model retained top contributing features based on gain.

---

##  Submission

- Final model trained on the full training data.
- Predictions made on test set were `expm1()` transformed and clipped to eliminate negative values.
- Final submission in `submission_filtered_features.csv`.

---

## 

 Environment

- Python 3.9+
- Libraries: `lightgbm`, `optuna`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

---

##  Future Improvements

- Try alternative models like **CatBoost** or **XGBoost**
- Consider ensemble approaches
- Evaluate temporal patterns more deeply (e.g., holidays, month)

---

## 👤 Author

Built collaboratively with assistance from ChatGPT as a data science partner.
