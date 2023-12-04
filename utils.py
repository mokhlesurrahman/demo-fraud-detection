""" Data Manipulation and Callbacks """
import datetime as dt
import numpy as np
import pandas as pd
from taipy.gui import State, navigate, notify
import xgboost as xgb
from shap import Explainer, Explanation
from sklearn.metrics import confusion_matrix

column_names = [
    "amt",
    "zip",
    "city_pop",
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

    first = state.transactions.iloc[idx]["first"]
    last = state.transactions.iloc[idx]["last"]

    state.specific_transactions = state.transactions[
        (state.transactions["first"] == first) & (state.transactions["last"] == last)
    ]

    state.selected_transaction = state.transactions.loc[[idx]]

    state.selected_client = f"{first} {last}"

    navigate(state, "Analysis")


def generate_transactions(
    state: State,
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    threshold: float,
    start_date="2020-06-21",
    end_date="2030-01-01",
) -> [pd.DataFrame, Explanation]:
    """
    Generates a DataFrame of transactions with the fraud prediction

    Args:
        - state: the state of the app
        - df: the DataFrame containing the transactions
        - model: the model used to predict the fraud
        - threshold: the threshold used to determine if a transaction is fraudulent
        - start_date: the start date of the transactions
        - end_date: the end date of the transactions

    Returns:
        - a DataFrame of transactions with the fraud prediction
    """
    start_date = str(start_date)
    end_date = str(end_date)
    start_date_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")
    # Make sure the dates are separated by at least one day
    if (end_date_dt - start_date_dt).days < 1:
        notify(state, "error", "The start date must be before the end date")
        raise Exception("The start date must be before the end date")
    # Make sure that start_date is between 2020-06-21 and 2020-06-30
    if not (dt.datetime(2020, 6, 21) <= start_date_dt <= dt.datetime(2020, 6, 30)):
        notify(
            state, "error", "The start date must be between 2020-06-21 and 2020-06-30"
        )
        raise Exception("The start date must be between 2020-06-21 and 2020-06-30")
    df["age"] = dt.date.today().year - pd.to_datetime(df["dob"]).dt.year
    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    df["day"] = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["trans_date_trans_time"]).dt.month
    test = df[
        [
            "category",
            "amt",
            "zip",
            "city_pop",
            "age",
            "hour",
            "day",
            "month",
            "is_fraud",
        ]
    ]
    test = pd.get_dummies(test, drop_first=True)
    test = test[df["trans_date_trans_time"].between(str(start_date), str(end_date))]

    X_test = test.drop("is_fraud", axis="columns")
    X_test_values = X_test.values

    transactions = df[
        df["trans_date_trans_time"].between(str(start_date), str(end_date))
    ]
    raw_results = model.predict(X_test_values)
    results = [str(min(1, round(result, 2))) for result in raw_results]
    transactions.insert(0, "fraud_value", results)
    # Low if under 0.2, Medium if under 0.5, High if over 0.5
    results = ["Low" if float(result) < 0.2 else "Medium" for result in raw_results]
    for i, result in enumerate(results):
        if result == "Medium" and float(raw_results[i]) > 0.5:
            results[i] = "High"
    transactions.insert(0, "fraud_confidence", results)
    results = [float(result) > threshold for result in raw_results]
    transactions.insert(0, "fraud", results)

    explainer = Explainer(model)
    sv = explainer(X_test)
    explaination = Explanation(sv, sv.base_values, X_test, feature_names=X_test.columns)
    # Drop Unnamed: 0 column if it exists
    if "Unnamed: 0" in transactions.columns:
        transactions = transactions.drop(columns=["Unnamed: 0"])
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
    results = [
        float(result) > threshold
        for result in state.original_transactions["fraud_value"]
    ]
    state.original_transactions["fraud"] = results
    state.original_transactions = state.original_transactions
    y_pred = results
    y_true = state.original_transactions["is_fraud"]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]

    dataset = state.original_transactions[:10000]
    state.true_positives = dataset[
        (dataset["is_fraud"] == True) & (dataset["fraud"] == True)
    ]
    state.true_negatives = dataset[
        (dataset["is_fraud"] == False) & (dataset["fraud"] == False)
    ]
    state.false_positives = dataset[
        (dataset["is_fraud"] == False) & (dataset["fraud"] == True)
    ]
    state.false_negatives = dataset[
        (dataset["is_fraud"] == True) & (dataset["fraud"] == False)
    ]

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
                "font": {"color": "white" if value < 0.5 else "black", "size": 30},
                "showarrow": False,
            }
            layout["annotations"].append(annotation)

    state.confusion_data = data
    state.confusion_layout = layout
    update_table(state)
    return (
        state.true_positives,
        state.true_negatives,
        state.false_positives,
        state.false_negatives,
    )


def update_table(state: State) -> None:
    """
    Updates the table of transactions displayed

    Args:
        - state: the state of the app
    """
    if state.selected_table == "True Positives":
        state.displayed_table = state.true_positives
    elif state.selected_table == "False Positives":
        state.displayed_table = state.false_positives
    elif state.selected_table == "True Negatives":
        state.displayed_table = state.true_negatives
    elif state.selected_table == "False Negatives":
        state.displayed_table = state.false_negatives
