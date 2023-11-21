import datetime as dt
import numpy as np
import pandas as pd
from taipy.gui import Gui
import xgboost as xgb
from shap import Explainer, Explanation
from sklearn.metrics import confusion_matrix

N = 10000
threshold = 0.5
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

df = pd.read_csv("fraudTrain.csv")

df["age"] = dt.date.today().year - pd.to_datetime(df["dob"]).dt.year
df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
df["day"] = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek
df["month"] = pd.to_datetime(df["trans_date_trans_time"]).dt.month
train = df[
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
train = pd.get_dummies(train, drop_first=True)

y_train = train["is_fraud"]
y_train_values = y_train.values
X_train = train.drop("is_fraud", axis="columns")
X_train_values = X_train.values

model = xgb.XGBRegressor()
model.fit(X_train_values, y_train_values)

explainer = Explainer(model)
sv = explainer(X_train[:N])
explaination = Explanation(sv, sv.base_values, X_train, feature_names=X_train.columns)

transactions = df[:N]
# Remove first column
transactions = transactions.drop("Unnamed: 0", axis="columns")
raw_results = model.predict(X_train_values[:N])
results = [str(round(result, 2)) for result in raw_results]
transactions.insert(0, "fraud_value", results)
results = [float(result) > threshold for result in raw_results]
# Put a column in first position with the results
transactions.insert(0, "fraud", results)

exp_data = pd.DataFrame({"Feature": [], "Influence": []})
fraud_text = "No row selected"


def explain_pred(state, var_name: str, payload: dict) -> None:
    """
    When a transaction is selected in the table
    Explain the prediction using SHAP, update the waterfall chart
    """
    idx = payload["index"]
    exp = explaination[idx]

    feature_values = [-value for value in list(exp.values)]
    data_values = list(exp.data)

    for i, value in enumerate(data_values):
        if type(value) is float:
            value = round(value, 2)
            data_values[i] = value

    names = [f"{name}: {value}" for name, value in zip(column_names, data_values)]

    exp_data = pd.DataFrame({"Feature": names, "Influence": feature_values})
    exp_data["abs_importance"] = exp_data["Influence"].abs()
    exp_data = exp_data.sort_values(by="abs_importance", ascending=False)
    exp_data = exp_data.drop(columns=["abs_importance"])
    exp_data = exp_data[:5]
    state.exp_data = exp_data

    if transactions.iloc[idx]["fraud"]:
        state.fraud_text = "Why is this transaction fraudulent?"
    else:
        state.fraud_text = "Why is this transaction not fraudulent?"


layout_dict = {
    "margin": {"b": 150},
}


page1 = """
<|layout|columns=3 2|
Select a row to explain the prediction
<|{transactions}|table|on_action=explain_pred|>

<|{fraud_text}|text|>
<|{exp_data}|chart|type=waterfall|x=Feature|y=Influence|layout={layout_dict}|>
|>
"""

# Create a list of amt values for fraudulent and non-fraudulent transactions
amt_fraud = transactions[transactions["fraud"]]["amt"]
amt_no_fraud = transactions[~transactions["fraud"]]["amt"]
amt_data = [
    {"Amount ($)": list(amt_no_fraud)[: len(amt_fraud)]},
    {"Amount ($)": list(amt_fraud)},
]

amt_options = [
    # First data set displayed as green-ish, and 5 bins
    {
        "marker": {"color": "#4A4", "opacity": 0.8},
        "nbinsx": 100,
    },
    # Second data set displayed as red-ish, and 25 bins
    {
        "marker": {"color": "#A33", "opacity": 0.8, "text": "Compare Data"},
        "nbinsx": 100,
    },
]

amt_layout = {
    # Overlay the two histograms
    "barmode": "overlay",
    "title": "Transaction Amount Distribution (Red = Fraudulent)",
    "showlegend": False,
}

male_fraud_percentage = len(
    transactions[transactions["fraud"]][transactions["gender"] == "M"]
) / len(transactions[transactions["fraud"]])
female_fraud_percentage = 1 - male_fraud_percentage
male_not_fraud_percentage = len(
    transactions[~transactions["fraud"]][transactions["gender"] == "M"]
) / len(transactions[~transactions["fraud"]])
female_not_fraud_percentage = 1 - male_not_fraud_percentage

gender_data = pd.DataFrame(
    {
        "Fraudulence": ["Not Fraud", "Fraud"],
        "Male": [male_not_fraud_percentage, male_fraud_percentage],
        "Female": [female_not_fraud_percentage, female_fraud_percentage],
    }
)

gender_layout = {"title": "Distribution of Fraud by Gender"}

categories = transactions["category"].unique()
fraud_categories = [
    len(transactions[transactions["fraud"]][transactions["category"] == category])
    for category in categories
]
fraud_categories_norm = [
    category / len(transactions[transactions["fraud"]]) for category in fraud_categories
]
not_fraud_categories = [
    len(transactions[~transactions["fraud"]][transactions["category"] == category])
    for category in categories
]
not_fraud_categories_norm = [
    category / len(transactions[~transactions["fraud"]])
    for category in not_fraud_categories
]
diff_categories = [
    fraud_categories_norm[i] - not_fraud_categories_norm[i]
    for i in range(len(categories))
]
cat_data = pd.DataFrame(
    {
        "Category": categories,
        "Difference": diff_categories,
    }
)

cat_data = cat_data.sort_values(by="Difference", ascending=False)
cat_layout = {"title": "Difference in Fraudulence by Category (Positive = Fraudulent)"}

age = range(111)
fraud_age = [
    len(transactions[transactions["fraud"]][transactions["age"] == age])
    / len(transactions[transactions["fraud"]])
    for age in age
]
not_fraud_age = [
    len(transactions[~transactions["fraud"]][transactions["age"] == age])
    / len(transactions[~transactions["fraud"]])
    for age in age
]
age_data = pd.DataFrame(
    {
        "Age": age,
        "Fraud": fraud_age,
        "Not Fraud": not_fraud_age,
    }
)
age_layout = {"title": "Distribution of Fraud by Age"}

hours = range(1, 25)
fraud_hours = [
    len(transactions[transactions["fraud"]][transactions["hour"] == hour])
    / len(transactions[transactions["fraud"]])
    for hour in hours
]
not_fraud_hours = [
    len(transactions[~transactions["fraud"]][transactions["hour"] == hour])
    / len(transactions[~transactions["fraud"]])
    for hour in hours
]
hour_data = pd.DataFrame(
    {
        "Hour": hours,
        "Fraud": fraud_hours,
        "Not Fraud": not_fraud_hours,
    }
)
hour_layout = {"title": "Distribution of Fraud by Hour"}

days = range(7)
days_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
fraud_days = [
    len(transactions[transactions["fraud"]][transactions["day"] == day])
    / len(transactions[transactions["fraud"]])
    for day in days
]
not_fraud_days = [
    len(transactions[~transactions["fraud"]][transactions["day"] == day])
    / len(transactions[~transactions["fraud"]])
    for day in days
]
day_data = pd.DataFrame(
    {
        "Day": days_names,
        "Fraud": fraud_days,
        "Not Fraud": not_fraud_days,
    }
)
day_layout = {"title": "Distribution of Fraud by Day"}

months = range(1, 13)
months_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
fraud_months = [
    len(transactions[transactions["fraud"]][transactions["month"] == month])
    / len(transactions[transactions["fraud"]])
    for month in months
]
not_fraud_months = [
    len(transactions[~transactions["fraud"]][transactions["month"] == month])
    / len(transactions[~transactions["fraud"]])
    for month in months
]
month_data = pd.DataFrame(
    {
        "Month": months_names,
        "Fraud": fraud_months,
        "Not Fraud": not_fraud_months,
    }
)
month_layout = {"title": "Distribution of Fraud by Month"}


page2 = """
<|{amt_data}|chart|type=histogram|options={amt_options}|layout={amt_layout}|>
<|{gender_data}|chart|type=bar|x=Fraudulence|y[1]=Male|y[2]=Female|layout={gender_layout}|>
<|{cat_data}|chart|type=bar|x=Category|y=Difference|orientation=v|layout={cat_layout}|>
<|{age_data}|chart|type=line|x=Age|y[1]=Not Fraud|y[2]=Fraud|layout={age_layout}|>
<|{hour_data}|chart|type=bar|x=Hour|y[1]=Not Fraud|y[2]=Fraud|layout={hour_layout}|>
<|{day_data}|chart|type=bar|x=Day|y[1]=Not Fraud|y[2]=Fraud|layout={day_layout}|>
<|{month_data}|chart|type=bar|x=Month|y[1]=Not Fraud|y[2]=Fraud|layout={month_layout}|>
"""

confusion_text = "Confusion Matrix"


def update_threshold(state):
    state.results = [float(result) > state.threshold for result in raw_results]
    state.transactions["fraud"] = results
    state.transactions = state.transactions

    y_pred = state.results
    y_true = y_train_values[:N]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    state.confusion_text = cm
    print(cm)


page3 = """
<|{threshold}|slider|min=0.0|max=1.0|on_change=update_threshold|>
<|{confusion_text}|text|>
"""

root = """
<|navbar|>
"""

pages = {
    "/": root,
    "page1": page1,
    "page2": page2,
    "page3": page3,
}

Gui(pages=pages).run()
