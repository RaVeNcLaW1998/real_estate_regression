
# 🏠 Real Estate Price Prediction App

This project is a Machine Learning web application that predicts **real estate property prices** based on input features such as property size, number of bedrooms, and more.

The app is powered by **Streamlit** and trained using **Linear Regression** from scikit-learn.

---

## 🚀 Live Demo

🔗 [Click here to open the app on Streamlit Cloud](https://realestateregression-ak.streamlit.app)
---

## 📂 Project Structure

```
real_estate_project/
├── app.py                      <- Streamlit web application
├── requirements.txt            <- Python dependencies
├── .gitignore                  <- Git ignore rules
├── README.md                   <- You're reading it!
│
├── data/
│   └── raw/
│       └── final.csv           <- Your dataset (NOT pushed to GitHub)
│
├── real_estate/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── train.py            <- Core training logic with logging & error handling
│       ├── predict.py
│       └── __init__.py
│
├── reports/
│   └── figures/                <- Future visualizations
│
├── tests/
│   └── test_train.py           <- Unit tests using pytest
│
└── real_estate/logs/
    └── train.log               <- Model training log file
```

---

## ✅ Features

- 🧠 Linear Regression Model (scikit-learn)
- 📈 Model Evaluation (MAE, R²)
- 📊 Streamlit Web Interface
- ✅ Logging & Error Handling
- 🧪 Unit Tests with Pytest
- 📁 Production-grade Folder Structure

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your dataset

Place `final.csv` in:

```
data/raw/final.csv
```

### 3. Train the model

```bash
python real_estate/modeling/train.py
```

### 4. Run the web app

```bash
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 💾 Logging

All logs are saved in:

```
real_estate/logs/train.log
```

---

## ☁️ Deploying on Streamlit Cloud

1. Push your code to a **public GitHub repository**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New App”** and connect your repo
4. Set `app.py` as the entry point
5. Add your `requirements.txt`
6. Click **Deploy**
---

## 🔒 License

This project is licensed under the MIT License.
