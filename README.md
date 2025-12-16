# 🏨 Hotel Booking Cancellation Prediction

**Kaggle Rank:** **26 / 1210**
Predict whether a customer will cancel their hotel booking using an end‑to‑end machine learning pipeline.

---

## 📸 Visuals

<img width="641" height="134" alt="Feature distributions" src="https://github.com/user-attachments/assets/f6c6849c-fbbc-4fa9-bb8e-75b281941af4" />

<img width="599" height="296" alt="Model performance comparison" src="https://github.com/user-attachments/assets/bd50d492-e8d9-4fb6-9cb9-26a660aa920a" />

🎥 **Video Walkthrough:** [https://drive.google.com/drive/folders/1gtRkcdEqWsN09ifX_5TdFoYZBjTd6IvB?usp=sharing](https://drive.google.com/drive/folders/1gtRkcdEqWsN09ifX_5TdFoYZBjTd6IvB?usp=sharing)

---

## 🔍 Project Overview

Accurately predicting booking cancellations helps hotels optimize inventory, reduce revenue loss, and improve customer engagement.

**Problem Statement:**
Given historical booking data, predict `booking_status` (**1 = cancelled**, **0 = not cancelled**) for new bookings.

**Real‑World Impact:**

* Reduce losses from last‑minute cancellations
* Improve overbooking and pricing strategies
* Enable targeted customer communication

---

## 📊 Dataset

Hotel booking records with the following features:

| Feature                | Description                      |
| ---------------------- | -------------------------------- |
| `id`                   | Unique booking identifier        |
| `adults`, `children`   | Number of guests                 |
| `weekends`, `weekdays` | Stay duration                    |
| `meal_type`            | Selected meal plan               |
| `room_type`            | Room category                    |
| `arrival`              | Arrival date                     |
| `lead_time`            | Days between booking and arrival |
| `segment`              | Booking segment                  |
| `repeat`               | Repeat customer flag             |
| `price`                | Average room price               |
| `requests`             | Special requests                 |
| `booking_status`       | Target variable                  |

**Files:**

* `train.csv` – Training data (labels included)
* `test.csv` – Test data (labels hidden)
* `sample_submission.csv` – Submission format

---

## 📈 Exploratory Data Analysis (EDA)

**Key Findings:**

* Most bookings involve **1–2 adults**; children are uncommon
* `lead_time` ranges **0–443 days** → capped at **365** for stability
* `price = 0` indicates free/erroneous entries → corrected
* Minor outliers in `weekdays` and `requests`

**Visuals Included:**

* Numerical distributions
* Cancellation rates by `meal_type` and `room_type`
* Correlation heatmap
* Outlier boxplots
* Category vs target countplots

---

## 🧹 Preprocessing

* Replaced **0 adults → 1** (invalid bookings)
* Capped `lead_time` at **365 days**
* Replaced **price = 0** with median
* Label encoding for categorical variables
* Standard scaling where required

---

## 🤖 Modeling & Tuning

**Baseline Models (10):**
Logistic Regression, Linear SVC, KNN, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM

**Top Performers:** Ensemble methods

**Hyperparameter Tuning:** RandomizedSearchCV on top 3 models

### 🏆 Best Model — Tuned XGBoost

* `n_estimators`: 437
* `max_depth`: 9
* `learning_rate`: 0.104
* `subsample`: 0.951
* `colsample_bytree`: 0.729

**Performance:**

| Model                 | Train Acc. |  Val. Acc. |
| --------------------- | ---------: | ---------: |
| Random Forest (tuned) |     0.9943 |     0.8839 |
| Extra Trees (tuned)   |     0.9943 |     0.8856 |
| **XGBoost (tuned)**   | **0.9378** | **0.8892** |

**Insight:** XGBoost delivers the best generalization with controlled overfitting.

---

## 🛠️ Tech Stack

* **Languages & Libraries:** Python, Pandas, NumPy, Scikit‑learn
* **Models:** XGBoost, LightGBM
* **Visualization:** Matplotlib, Seaborn

---

## 📌 Summary

**Ranked 26 / 1210 on Kaggle**, this project demonstrates a complete ML workflow—from EDA and preprocessing to ensemble modeling and tuning—focused on a real business problem with measurable impact.
