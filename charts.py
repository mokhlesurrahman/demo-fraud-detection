""" Prepare data for charts """

import pandas as pd


def gen_amt_data(transactions: pd.DataFrame) -> list:
    """
    Create a list of amt values for fraudulent and non-fraudulent transactions

    Args:
        - transactions: the transactions dataframe

    Returns:
        - a list of two dictionaries containing the data for the two histograms
    """
    amt_fraud = transactions[transactions["fraud"]]["amt"]
    amt_no_fraud = transactions[~transactions["fraud"]]["amt"]
    amt_data = [
        {"Amount ($)": list(amt_no_fraud)},
        {"Amount ($)": list(amt_fraud)},
    ]
    return amt_data


def gen_gender_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe containing the percentage of fraudulent transactions
    per gender

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
    male_fraud_percentage = len(
        transactions[transactions["fraud"]].loc[transactions["gender"] == "M"]
    ) / len(transactions[transactions["fraud"]])
    female_fraud_percentage = 1 - male_fraud_percentage
    male_not_fraud_percentage = len(
        transactions[~transactions["fraud"]].loc[transactions["gender"] == "M"]
    ) / len(transactions[~transactions["fraud"]])
    female_not_fraud_percentage = 1 - male_not_fraud_percentage

    gender_data = pd.DataFrame(
        {
            "Fraudulence": ["Not Fraud", "Fraud"],
            "Male": [male_not_fraud_percentage, male_fraud_percentage],
            "Female": [female_not_fraud_percentage, female_fraud_percentage],
        }
    )
    return gender_data


def gen_cat_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe with the percentage difference
    between fraudulent and non-fraudulent transactions per category

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
    categories = transactions["category"].unique()
    fraud_categories = [
        len(
            transactions[transactions["fraud"]].loc[
                transactions["category"] == category
            ]
        )
        for category in categories
    ]
    fraud_categories_norm = [
        category / len(transactions[transactions["fraud"]])
        for category in fraud_categories
    ]
    not_fraud_categories = [
        len(
            transactions[~transactions["fraud"]].loc[
                transactions["category"] == category
            ]
        )
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
    return cat_data


def gen_age_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe with the percentage of fraudulent transactions
    per age

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
    age = range(111)
    fraud_age = [
        len(transactions[transactions["fraud"]].loc[transactions["age"] == age])
        / len(transactions[transactions["fraud"]])
        for age in age
    ]
    not_fraud_age = [
        len(transactions[~transactions["fraud"]].loc[transactions["age"] == age])
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
    return age_data


def gen_hour_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe with the percentage of fraudulent transactions
    per hour

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
    hours = range(1, 25)
    fraud_hours = [
        len(transactions[transactions["fraud"]].loc[transactions["hour"] == hour])
        / len(transactions[transactions["fraud"]])
        for hour in hours
    ]
    not_fraud_hours = [
        len(transactions[~transactions["fraud"]].loc[transactions["hour"] == hour])
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
    return hour_data


def gen_day_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe with the percentage of fraudulent transactions
    per weekday

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
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
        len(transactions[transactions["fraud"]].loc[transactions["day"] == day])
        / len(transactions[transactions["fraud"]])
        for day in days
    ]
    not_fraud_days = [
        len(transactions[~transactions["fraud"]].loc[transactions["day"] == day])
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
    return day_data


def gen_month_data(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataframe with the percentage of fraudulent transactions
    per month

    Args:
        - transactions: the transactions dataframe

    Returns:
        - the resulting dataframe
    """
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
        len(transactions[transactions["fraud"]].loc[transactions["month"] == month])
        / len(transactions[transactions["fraud"]])
        for month in months
    ]
    not_fraud_months = [
        len(transactions[~transactions["fraud"]].loc[transactions["month"] == month])
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
    return month_data
