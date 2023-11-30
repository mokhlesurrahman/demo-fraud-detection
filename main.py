""" Fraud Detection App """
import pickle

import numpy as np
import pandas as pd
from taipy.gui import Gui, Icon, State, navigate, notify

from utils import (
    explain_pred,
    generate_transactions,
    update_threshold,
)
from charts import *

DATA_POINTS = 10000
threshold = "0.5"
threshold_lov = np.arange(0, 1, 0.01)
confusion_text = "Confusion Matrix"
fraud_text = "No row selected"
exp_data = pd.DataFrame({"Feature": [], "Influence": []})

df = pd.read_csv("data/fraud_data.csv")
df["merchant"] = df["merchant"].str[6:]
model = pickle.load(open("model.pkl", "rb"))
transactions, explaination = generate_transactions(df, model, float(threshold))
specific_transactions = transactions
selected_client = "No client selected"
start_date = "2020-06-21"
end_date = "2020-06-22"


def fraud_style(_: State, index: int, values: list) -> str:
    """
    Style the transactions table: red if fraudulent

    Args:
        - state: the state of the app
        - index: the index of the row

    Returns:
        - the style of the row
    """
    if values.iloc[1] == "High":
        return "red-row"
    elif values.iloc[1] == "Medium":
        return "orange-row"
    return ""


amt_data = gen_amt_data(transactions)
gender_data = gen_gender_data(transactions)
cat_data = gen_cat_data(transactions)
age_data = gen_age_data(transactions)
hour_data = gen_hour_data(transactions)
day_data = gen_day_data(transactions)
month_data = gen_month_data(transactions)

transactions = transactions[:DATA_POINTS]


waterfall_layout = {
    "margin": {"b": 150},
}

amt_options = [
    {
        "marker": {"color": "#4A4", "opacity": 0.8},
        "xbins": {"start": 0, "end": 2000, "size": 10},
        "histnorm": "probability",
    },
    {
        "marker": {"color": "#A33", "opacity": 0.8, "text": "Compare Data"},
        "xbins": {"start": 0, "end": 2000, "size": 10},
        "histnorm": "probability",
    },
]

amt_layout = {
    "barmode": "overlay",
    "showlegend": True,
}

confusion_data = pd.DataFrame({"Predicted": [], "Actual": [], "Values": []})
confusion_layout = None
confusion_options = {"colorscale": "YlOrRd", "displayModeBar": False}
confusion_config = {"scrollZoom": False, "displayModeBar": False}

transactions = df
transactions = transactions.drop("Unnamed: 0", axis="columns")


def on_init(state: State) -> None:
    """
    Generate the confusion matrix on start

    Args:
        - state: the state of the app
    """
    # update_threshold(state)


def update_transactions(state: State) -> None:
    """
    Detects frauds in the selected time period

    Args:
        - state: the state of the app
    """
    notify(state, "info", "Predicting fraud...")
    state.transactions, state.explaination = generate_transactions(
        df, model, float(state.threshold), state.start_date, state.end_date
    )
    number_of_fraud = len(state.transactions[state.transactions["fraud"] == True])
    notify(state, "success", f"Predicted {number_of_fraud} fraudulent transactions")


menu_lov = [
    ("Transactions", Icon("images/transactions.png", "Transactions")),
    ("Analysis", Icon("images/analysis.png", "Analysis")),
    ("Fraud Distribution", Icon("images/distribution.png", "Fraud Distribution")),
    ("Threshold Selection", Icon("images/threshold.png", "Threshold Selection")),
]

page = "Transactions"


def menu_fct(state, var_name, var_value):
    """Function that is called when there is a change in the menu control."""
    state.page = var_value["args"][0]
    navigate(state, state.page.replace(" ", "-"))


ROOT = """
<|menu|label=Menu|lov={menu_lov}|on_action=menu_fct|>
"""

TRANSACTIONS_PAGE = """
# List of **Transactions**{: .color-primary}

--------------------------------------------------------------------

## Select start and end date for a prediction
<|layout|columns=1 1 3|
Start Date: <|{start_date}|date|>

End Date: <|{end_date}|date|>
|>

<|Detect Frauds|button|on_action=update_transactions|>

<|{transactions}|table|on_action=explain_pred|style=fraud_style|filter|rebuild|>
"""

ANALYSIS_PAGE = """
<|layout|columns=2 3|
<|card|
## <|{fraud_text}|text|>
<|{exp_data}|chart|type=waterfall|x=Feature|y=Influence|layout={waterfall_layout}|>
|>

<|
## Transactions of client: **<|{selected_client}|text|raw|>**{: .color-primary}
<|{specific_transactions}|table|style=fraud_style|filter|on_action=explain_pred|>
|>
|>
"""

CHART_PAGE = """
# Fraud **Distribution**{: .color-primary}

--------------------------------------------------------------------

## Charts of fraud distribution by feature

<|{amt_data}|chart|type=histogram|title=Transaction Amount Distribution|color[2]=red|color[1]=green|name[2]=Fraud|name[1]=Not Fraud|options={amt_options}|layout={amt_layout}|>
<br/><|{gender_data}|chart|type=bar|x=Fraudulence|y[1]=Male|y[2]=Female|title=Distribution of Fraud by Gender|>
<br/><|{cat_data}|chart|type=bar|x=Category|y=Difference|orientation=v|title=Difference in Fraudulence by Category (Positive = Fraudulent)|>
<br/><|{hour_data}|chart|type=bar|x=Hour|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Hour|>
<br/><|{day_data}|chart|type=bar|x=Day|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Day|>
"""

THRESHOLD_PAGE = """
# Threshold **Selection**{: .color-primary}

--------------------------------------------------------------------

## Select a threshold of confidence to filter the transactions
<|{threshold}|slider|on_change=update_threshold|lov=0.001;0.005;0.01;0.05;0.1;0.5|>
<|{confusion_data}|chart|type=heatmap|z=Values|x=Predicted|y=Actual|layout={confusion_layout}|options={confusion_options}|plot_config={confusion_config}|height=70vh|>
"""

pages = {
    "/": ROOT,
    "Transactions": TRANSACTIONS_PAGE,
    "Analysis": ANALYSIS_PAGE,
    "Fraud-Distribution": CHART_PAGE,
    "Threshold-Selection": THRESHOLD_PAGE,
}

Gui(pages=pages).run(title="Fraud Detection Demo", dark_mode=False)
