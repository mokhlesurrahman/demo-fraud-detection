import datetime as dt
import pandas as pd
import xgboost as xgb
from taipy.gui import Gui
from shap import Explainer, Explanation

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

model = xgb.XGBClassifier()
model.fit(X_train_values, y_train_values)

explainer = Explainer(model)
sv = explainer(X_train[:10000])
explaination = Explanation(sv, sv.base_values, X_train, feature_names=X_train.columns)
exp_data = pd.DataFrame({"Feature": [], "Influence": []})
transactions = X_train[:10000]


def explain_pred(state, var_name: str, payload: dict) -> None:
    """
    When a transaction is selected in the table
    Explain the prediction using SHAP, update the waterfall chart
    """
    idx = payload["index"]
    exp = explaination[idx]
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


layout_dict = {
    "margin": {"b": 150},
}


page = """
<|layout|columns=2 1|
<|{transactions}|table|on_action=explain_pred|>

<|{exp_data}|chart|type=waterfall|x=Feature|y=Influence|layout={layout_dict}|>
|>
"""

Gui(page).run()
