""" Data Manipulation and Callbacks """
import datetime as dt
import numpy as np
import pandas as pd
from taipy.gui import State
import xgboost as xgb
from shap import Explainer, Explanation
from sklearn.metrics import confusion_matrix

column_names = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "age",
    "hour",
    "day",
    "month",
    "category_food_dining",
    "category_gas_transport",
    "category_grocery_net",
    "category_grocery_pos",
    "category_health_fitness",
    "category_home",
    "category_kids_pets",
    "category_misc_net",
    "category_misc_pos",
    "category_personal_care",
    "category_shopping_net",
    "category_shopping_pos",
    "category_travel",
]


def explain_pred(state: State, _: str, payload: dict) -> None:
    """
    When a transaction is selected in the table
    Explain the prediction using SHAP, update the waterfall chart

    Args:
        - state: the state of the app
        - payload: the payload of the event containing the index of the transaction
    """
    idx = payload["index"]
    exp = state.explaination[idx]

    feature_values = [-value for value in list(exp.values)]
    data_values = list(exp.data)

    for i, value in enumerate(data_values):
        if isinstance(value, float):
            value = round(value, 2)
            data_values[i] = value

    names = [f"{name}: {value}" for name, value in zip(column_names, data_values)]

    exp_data = pd.DataFrame({"Feature": names, "Influence": feature_values})
    exp_data["abs_importance"] = exp_data["Influence"].abs()
    exp_data = exp_data.sort_values(by="abs_importance", ascending=False)
    exp_data = exp_data.drop(columns=["abs_importance"])
    exp_data = exp_data[:5]
    state.exp_data = exp_data

    if state.transactions.iloc[idx]["fraud"]:
        state.fraud_text = "Why is this transaction fraudulent?"
    else:
        state.fraud_text = "Why is this transaction not fraudulent?"


def generate_transactions(
    df: pd.DataFrame, model: xgb.XGBRegressor, threshold: float
) -> [pd.DataFrame, Explanation]:
    """
    Generates a DataFrame of transactions with the fraud prediction

    Args:
        - df: the DataFrame containing the transactions
        - model: the model used to predict the fraud
        - threshold: the threshold used to determine if a transaction is fraudulent


    Returns:
        - a DataFrame of transactions with the fraud prediction
    """
    df["age"] = dt.date.today().year - pd.to_datetime(df["dob"]).dt.year
    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    df["day"] = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["trans_date_trans_time"]).dt.month
    test = df[
        [
            "category",
            "amt",
            "zip",
            "lat",
            "long",
            "city_pop",
            "merch_lat",
            "merch_long",
            "age",
            "hour",
            "day",
            "month",
            "is_fraud",
        ]
    ]
    test = pd.get_dummies(test, drop_first=True)

    X_test = test.drop("is_fraud", axis="columns")
    X_test_values = X_test.values

    transactions = df
    transactions = transactions.drop("Unnamed: 0", axis="columns")
    raw_results = model.predict(X_test_values)
    results = [str(round(result, 2)) for result in raw_results]
    transactions.insert(0, "fraud_value", results)
    results = [float(result) > threshold for result in raw_results]
    transactions.insert(0, "fraud", results)

    explainer = Explainer(model)
    sv = explainer(X_test)
    explaination = Explanation(sv, sv.base_values, X_test, feature_names=X_test.columns)

    return transactions, explaination


def update_threshold(state: State) -> None:
    """
    Change the threshold used to determine if a transaction is fraudulent
    Generate the confusion matrix

    Args:
        - state: the state of the app
    """
    threshold = float(state.threshold)
    results = [
        float(result) > threshold for result in state.transactions["fraud_value"]
    ]
    state.transactions["fraud"] = results
    state.transactions = state.transactions
    y_pred = results
    y_true = state.transactions["is_fraud"]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]

    data = {
        "Values": [
            [fn, tp],
            [tn, fp],
        ],
        "Actual": ["Fraud", "Not Fraud"],
        "Predicted": ["Not Fraud", "Fraud"],
    }

    layout = {
        "annotations": [],
        "xaxis": {"ticks": "", "side": "top"},
        "yaxis": {"ticks": "", "ticksuffix": " "},
    }

    predicted = data["Predicted"]
    actuals = data["Actual"]
    for actual, _ in enumerate(actuals):
        for pred, _ in enumerate(predicted):
            value = data["Values"][actual][pred]
            annotation = {
                "x": predicted[pred],
                "y": actuals[actual],
                "text": f"{str(round(value, 3)*100)[:4]}%",
                "font": {"color": "white" if value < 0.5 else "black"},
                "showarrow": False,
            }
            layout["annotations"].append(annotation)

    state.confusion_data = data
    state.confusion_layout = layout
