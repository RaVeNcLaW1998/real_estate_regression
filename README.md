
# Real Estate Price Prediction Project

This project builds a linear regression model to predict real estate prices based on property features.

## 📁 Project Structure

```
├── data/
│   └── raw/                  <- Place your original `final.csv` file here
├── real_estate/
│   ├── modeling/
│   │   ├── train.py          <- Main training pipeline with logging and error handling
├── tests/
│   └── test_train.py         <- Unit tests for data loading and feature splitting
```

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Place your `final.csv` file in `data/raw/`.

3. Run the training script:

```
python real_estate/modeling/train.py
```

4. Run tests:

```
pytest tests/
```

## 🛠 Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- pytest

## 📄 Logs

Logs are stored in `real_estate/logs/train.log`.

## 🔒 License

MIT License
