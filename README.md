# PhoneAddictionPredictor
A machine learning web application that predicts the phone addiction level of teenagers on a scale of 0–10 based on behavioral, psychological, and demographic features. Built with scikit-learn for modeling and Streamlit for the web interface.

# 📱 Teen Phone Addiction Level Predictor

A machine learning web application that predicts the **phone addiction level** of teenagers on a scale from 0 to 10, based on behavioral, psychological, and demographic inputs.

Built using **scikit-learn** for modeling and **Streamlit** for the interactive web interface.

---

## 🧠 Overview

This project uses a dataset of teenage individuals containing over 25 features, including:

- Age
- Daily phone usage hours
- Sleep hours
- Academic performance
- Social life rating
- Exercise habits
- Anxiety level
- Phone usage purpose
- And more...

The model outputs a numeric **addiction score** which reflects the predicted intensity of phone dependency.

---

## 🚀 Features

- 📊 Predicts phone addiction level on a 0–10 scale
- 📉 Linear Regression model trained with MAE ≈ 0.6
- 🔄 Handles categorical features using one-hot encoding (`pd.get_dummies`)
- 🖥️ Built-in web UI using Streamlit for easy interaction
- 💾 Trained model serialized and reused with `joblib`

---

## 🛠 Tech Stack

- Python
- pandas, numpy
- scikit-learn
- Streamlit
- joblib

---


