from taipy.gui import Gui
import pickle
import pandas as pd

# Load exp
with open("exp.pkl", "rb") as f:
    exp = pickle.load(f)

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

# Use plotly layout dict
page = """
## Why was this transaction not flagged as fraud
<|{exp_data}|chart|type=waterfall|x=Feature|y=Influence|>
"""

Gui(page).run()
