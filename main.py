""" Fraud Detection App """
import pickle

import numpy as np
import pandas as pd
from taipy.gui import Gui

from utils import explain_pred, generate_transactions, update_threshold
from charts import *

DATA_POINTS = 10000
threshold = "0.5"
threshold_lov = np.arange(0, 1, 0.01)
confusion_text = "Confusion Matrix"
fraud_text = "No row selected"
exp_data = pd.DataFrame({"Feature": [], "Influence": []})

df = pd.read_csv("data/fraudTest.csv")[:DATA_POINTS]
model = pickle.load(open("model.pkl", "rb"))
transactions, explaination = generate_transactions(df, model, float(threshold))

amt_data = gen_amt_data(transactions)
gender_data = gen_gender_data(transactions)
cat_data = gen_cat_data(transactions)
age_data = gen_age_data(transactions)
hour_data = gen_hour_data(transactions)
day_data = gen_day_data(transactions)
month_data = gen_month_data(transactions)


waterfall_layout = {
    "margin": {"b": 150},
}

amt_options = [
    {
        "marker": {"color": "#4A4", "opacity": 0.8},
        "nbinsx": 100,
    },
    {
        "marker": {"color": "#A33", "opacity": 0.8, "text": "Compare Data"},
        "nbinsx": 100,
    },
]

amt_layout = {
    # Overlay the two histograms
    "barmode": "overlay",
    "showlegend": False,
}

ROOT = """
<|navbar|>
"""

TRANSACTIONS_PAGE = """
<|layout|columns=3 2|
Select a row to explain the prediction
<|{transactions}|table|on_action=explain_pred|>

<|{fraud_text}|text|>
<|{exp_data}|chart|type=waterfall|x=Feature|y=Influence|layout={waterfall_layout}|>
|>
"""

CHART_PAGE = """
<|{amt_data}|chart|type=histogram|title=Transaction Amount Distribution (Red = Fraudulent)|options={amt_options}|layout={amt_layout}|>
<|{gender_data}|chart|type=bar|x=Fraudulence|y[1]=Male|y[2]=Female|title=Distribution of Fraud by Gender|>
<|{cat_data}|chart|type=bar|x=Category|y=Difference|orientation=v|title=Difference in Fraudulence by Category (Positive = Fraudulent)|>
<|{age_data}|chart|type=line|x=Age|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Age|>
<|{hour_data}|chart|type=bar|x=Hour|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Hour|>
<|{day_data}|chart|type=bar|x=Day|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Day|>
<|{month_data}|chart|type=bar|x=Month|y[1]=Not Fraud|y[2]=Fraud|title=Distribution of Fraud by Month|>
"""

THRESHOLD_PAGE = """
<|{threshold}|slider|on_change=update_threshold|lov=0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9|>
<|{confusion_text}|text|>
"""

pages = {
    "/": ROOT,
    "Transactions": TRANSACTIONS_PAGE,
    "Data-Visualization": CHART_PAGE,
    "Threshold-Selection": THRESHOLD_PAGE,
}

Gui(pages=pages).run()
