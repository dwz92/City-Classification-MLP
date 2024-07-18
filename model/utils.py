'''
Author: Jesse Xu
Contact: Jesse_Xu@live.com
'''
import numpy as np
import re
import pandas as pd


def to_numeric(s):
    """
    Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """
    Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """
    Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """
    Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """
    Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """
    Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def lab_to_num(label):
    if label == 'Dubai':
        return 0
    elif label == 'Rio de Janeiro':
        return 1
    elif label == 'New York City':
        return 2
    else:
        return 3

def preprocess_data_for_training(file_name, catagories=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]) -> pd.DataFrame:
    df = pd.read_csv(file_name)

    new_names = []

    #Handle target
    df['Label'] = df['Label'].apply(lab_to_num)

    # Handle Q1, Q2, Q3, Q4 (on a scale of 1-5)
    for cat in ["Q1", "Q2", "Q3", "Q4"]:
        if cat in catagories:
            df[cat] = df[cat].apply(get_number)
            df[cat] = pd.Categorical(df[cat], categories=[-1, 1, 2, 3, 4, 5])
            indicators = pd.get_dummies(df[cat], prefix=cat)
            new_names.extend(indicators.columns)
            df = pd.concat([df, indicators], axis=1)
            df.drop([cat], axis=1, inplace=True)

    # Handle Q5 (multi-category indicators)
    if "Q5" in catagories:
        for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
            cat_name = f"Q5_{cat}"
            new_names.append(cat_name)
            df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))
        df.drop(["Q5"], axis=1, inplace=True)

    # Handle Q6 (area rank categories)
    if "Q6" in catagories:
        df["Q6"] = df["Q6"].apply(get_number_list_clean)
        temp_names = []
        for i in range(1, 7):
            col_name = f"Q6_rank_{i}"
            temp_names.append(col_name)
            df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))
        df.drop(["Q6"], axis=1, inplace=True)
        for col in temp_names:
            df[col] = pd.Categorical(df[col], categories=[-1, 1, 2, 3, 4, 5, 6])
            indicators = pd.get_dummies(df[col], prefix=col)
            new_names.extend(indicators.columns)
            df = pd.concat([df, indicators], axis=1)
            df.drop([col], axis=1, inplace=True)

    # Handle Q7, Q8, Q9 (numerics)
    for cat in ["Q7", "Q8", "Q9"]:
        if cat in catagories:
            df[cat] = df[cat].apply(to_numeric).fillna(0)
            new_names.append(cat)

    return df[new_names + ["Label"]]

def preprocess_data_for_prediction(file_name, catagories=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]) -> pd.DataFrame:
    df = pd.read_csv(file_name)

    new_names = []

    # Handle Q1, Q2, Q3, Q4 (on a scale of 1-5)
    for cat in ["Q1", "Q2", "Q3", "Q4"]:
        if cat in catagories:
            df[cat] = df[cat].apply(get_number)
            df[cat] = pd.Categorical(df[cat], categories=[-1, 1, 2, 3, 4, 5])
            indicators = pd.get_dummies(df[cat], prefix=cat)
            new_names.extend(indicators.columns)
            df = pd.concat([df, indicators], axis=1)
            df.drop([cat], axis=1, inplace=True)

    # Handle Q5 (multi-category indicators)
    if "Q5" in catagories:
        for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
            cat_name = f"Q5_{cat}"
            new_names.append(cat_name)
            df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))
        df.drop(["Q5"], axis=1, inplace=True)

    # Handle Q6 (area rank categories)
    if "Q6" in catagories:
        df["Q6"] = df["Q6"].apply(get_number_list_clean)
        temp_names = []
        for i in range(1, 7):
            col_name = f"Q6_rank_{i}"
            temp_names.append(col_name)
            df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))
        df.drop(["Q6"], axis=1, inplace=True)
        for col in temp_names:
            df[col] = pd.Categorical(df[col], categories=[-1, 1, 2, 3, 4, 5, 6])
            indicators = pd.get_dummies(df[col], prefix=col)
            new_names.extend(indicators.columns)
            df = pd.concat([df, indicators], axis=1)
            df.drop([col], axis=1, inplace=True)

    # Handle Q7, Q8, Q9 (numerics)
    for cat in ["Q7", "Q8", "Q9"]:
        if cat in catagories:
            df[cat] = df[cat].apply(to_numeric).fillna(0)
            new_names.append(cat)

    return df[new_names]

def split_data(data: pd.DataFrame, n_train=1200, random_state=42):
    data = data.sample(frac=1, random_state=random_state)

    x = data.drop("Label", axis=1).values
    # y = pd.get_dummies(data["Label"].values)
    y = pd.get_dummies(data["Label"]).astype(int).values

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    return x_train, y_train, x_test, y_test