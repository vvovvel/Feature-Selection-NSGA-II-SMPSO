import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_german_data():
    """
    Pobiera zbiór German Credit (1000 rekordów, 20 atrybutów).
    Zwraca: X_train, X_test, y_train, y_test przygotowane pod KNN.
    """
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features
    y = dataset.data.targets

    # Mapowanie klas na 0 (zły) i 1 (dobry)
    y = y.iloc[:, 0].map({1: 1, 2: 0})

    return _split_and_meta(X, y)


def load_australian_data():
    """
    Pobiera zbiór Australian Credit (690 rekordów, 14 atrybutów).
    Zwraca: X_train, X_test, y_train, y_test przygotowane pod KNN.
    """
    dataset = fetch_ucirepo(id=143)
    X = dataset.data.features
    y = dataset.data.targets

    y = y.iloc[:, 0]

    return _split_and_meta(X, y)


def _split_and_meta(X, y):
    X_processed = X.copy()

    le = LabelEncoder()

    categorical_cols = X_processed.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    X_processed = X_processed.astype(float)

    X_processed = X_processed.fillna(X_processed.median())

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.30, random_state=42)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test