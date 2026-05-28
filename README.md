# ⚽ FIFA World Cup 2026 Match Prediction Engine

An advanced, machine learning-driven web application built with **Python**, **Streamlit**, and **Scikit-Learn** designed to simulate and predict match outcomes for the upcoming FIFA World Cup 2026. The application leverages a premium glassmorphic dark-themed interface to provide real-time outcome probabilities and precise goal scoring metrics based on live FIFA analytical point indexes.

Live App Link: `https://wc2026-prediction-model.streamlit.app`

---

## 🚀 Key Features

* **AI Match Simulation:** Simulates full match dynamics between any two FIFA-registered nations.
* **Exact Score Regression:** Computes distinct, realistic goal projections for both home and away venue configurations using specialized predictive models.
* **Win Probability Spectrum:** Displays a clean, interactive distribution strip representing the probability vector of a Home Win, Draw, or Away Win.
* **Team Analytical Profiles:** Displays current FIFA ranks and total global index points dynamically for the chosen match up.
* **Premium User Interface:** Designed with a modern, glassmorphic dark UI tailored for both desktop environments and standard layouts.

---

## 🛠️ Tech Stack & Architecture

* **Frontend Framework:** [Streamlit](https://streamlit.io/) (Data-centric Web UI framework)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)
* **Machine Learning Framework:** [Scikit-Learn](https://scikit-learn.org/)
  * **Classification Core:** `Random Forest Classifier` for determining the categorical match result probabilities.
  * **Regression Core:** Dual `Random Forest Regressors` tuned to approximate independent goal tallies ($G_{\text{home}}$ and $G_{\text{away}}$).

---

## 📁 Repository Structure

```text
├── app.py                  # Main Streamlit web application script
├── fifa_rankings.csv       # Historical and active dataset containing global points/ranks
├── classifier.pkl         # Trained Random Forest classifier model for probabilities
├── home_goal_model.pkl     # Trained Regressor estimating home team goal output
├── away_goal_model.pkl     # Trained Regressor estimating away team goal output
└── README.md               # Repository documentation
