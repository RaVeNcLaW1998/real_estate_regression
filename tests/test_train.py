import pandas as pd
from real_estate.modeling.train import load_data, split_features_target


def test_load_data():
    df = load_data("data/raw/final.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_split_features_target():
    df = pd.DataFrame(
        {"sqft": [1000, 1500], "bedrooms": [2, 3], "price": [300000, 450000]}
    )
    X, y = split_features_target(df, "price")
    assert "price" not in X.columns
    assert y.name == "price"
