# Hotel Booking Cancellation Prediction

RANK: 26/1210


<img width="641" height="134" alt="image" src="https://github.com/user-attachments/assets/f6c6849c-fbbc-4fa9-bb8e-75b281941af4" />

<img width="599" height="296" alt="image" src="https://github.com/user-attachments/assets/bd50d492-e8d9-4fb6-9cb9-26a660aa920a" />



Video Walkthrough: https://drive.google.com/drive/folders/1gtRkcdEqWsN09ifX_5TdFoYZBjTd6IvB?usp=sharing


## Project Overview

This project aims to predict whether a customer will cancel their hotel booking using machine learning techniques. Accurately predicting cancellations can help hotels optimize resource allocation, reduce revenue loss, and improve customer management.

**Problem Statement:**
Given historical booking data, predict the `booking_status` (1 = cancelled, 0 = not cancelled) for new bookings.

**Real-world Application:**

* Reducing financial loss due to last-minute cancellations.
* Optimizing room inventory and overbooking strategies.
* Improving personalized customer communication and retention.

---

## Dataset

The dataset contains hotel booking records with the following features:

| Column           | Description                                |
| ---------------- | ------------------------------------------ |
| `id`             | Unique booking identifier                  |
| `adults`         | Number of adults in the booking            |
| `children`       | Number of children in the booking          |
| `weekends`       | Number of weekend days in the stay         |
| `weekdays`       | Number of weekdays in the stay             |
| `meal_type`      | Selected meal plan                         |
| `room_type`      | Type of room booked                        |
| `arrival`        | Arrival date                               |
| `lead_time`      | Number of days between booking and arrival |
| `segment`        | Booking segment or type                    |
| `repeat`         | Whether the customer is a repeat customer  |
| `price`          | Average price per room                     |
| `requests`       | Number of special requests made            |
| `booking_status` | Target variable: 1 if cancelled, 0 if not  |

Files included:

* `train.csv` – Training dataset with target labels
* `test.csv` – Test dataset (target hidden)
* `sample_submission.csv` – Sample submission file

---

## Exploratory Data Analysis (EDA)

Key observations from the dataset:

* Most bookings include 1–2 adults; children are rare.
* Lead times range from 0 to 443 days; extremely high values capped at 365 for model stability.
* `price = 0` indicates free or erroneous bookings; these rows were handled during preprocessing.
* Minor outliers exist in `weekdays` and `requests`.

Visualizations included:

1. Distribution of numerical features (`adults`, `children`, `lead_time`, `price`).
2. Cancellation rates by `meal_type` and `room_type`.
3. Correlation heatmap for numerical features.
4. Boxplots to detect outliers in `lead_time` and `price`.
5. Countplots for categorical features vs target.

---

## Preprocessing

* Replaced `0 adults` with `1` since bookings without adults are invalid.
* Capped `lead_time` at 365 days to reduce skew.
* Replaced `price = 0` with median price to handle anomalies.
* Label encoding for categorical features like `meal_type`, `room_type`, `segment`.
* Standard scaling applied for linear models where needed.

---

## Model Selection & Hyperparameter Tuning

* 10 Baseline models tested: Logistic Regression, Linear SVC, KNN, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM.
* Ensembles (Random Forest, Extra Trees, XGBoost) performed best.
* RandomizedSearchCV used for hyperparameter tuning of top 3 models.

**Best Model:**

* **Tuned XGBoost**

  * n_estimators = 437
  * max_depth = 9
  * learning_rate = 0.104
  * subsample = 0.951
  * colsample_bytree = 0.729

**Performance Summary:**

| Model         | Train Accuracy | Validation Accuracy |
| ------------- | -------------- | ------------------- |
| Random Forest | 0.9943         | 0.8839 (tuned)      |
| Extra Trees   | 0.9943         | 0.8856 (tuned)      |
| XGBoost       | 0.9378         | 0.8892 (tuned)      |

**Insight:** Tuned XGBoost achieves the best generalization with controlled overfitting.

---

## Technologies & Libraries

* Python, Pandas, NumPy, Scikit-learn
* XGBoost, LightGBM
* Matplotlib, Seaborn
