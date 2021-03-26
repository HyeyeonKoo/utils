#-*-coding:utf-8-*-

"""
Module for preprocessing the stroke data.

Methods

    preprocess() : Main class for preprocessing the data.
        - data_path : Input data file path.
        - save_path : Output path. Only directory, Not file.
    
    process_by_column() : Preprocessing for each column in data.
        - data : Whole dataframe from pandas.

    to_categorical() : If column data can be splitted by few category or class, change to number.
        - data : Dataframe's column.
        - replace_nan : If specific data is Nan, replace with this.

    normalization() : If column data is numeric, normalize the data.
        - data : Dataframe's column.
        - replace_nan : If specific data is Nan, replace with this.

    fill_with_ratio() : If column data can be splitted by few category or class,
        it cannot be filled out by average, min, max etc.
        So, fill it with original data's ratio without unnecessary data.
        - data : Dataframe's column.
        - remove : Unnecessary data's name.
"""

import argparse
import pandas as pd
import random
from copy import deepcopy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def to_categorical(data, replace_nan=None):
    if data.isna().sum() > 0:
        data = data.fillna(replace_nan)

    category = list(set(data))
    result = pd.DataFrame(columns=[each for each in category])
    
    for i in range(len(data)):

        one_hot = [0] * len(category)
        one_hot[category.index(data.loc[i])] = 1
        result.loc[i] = one_hot

    return category, result


def normalization(data, replace_nan=None):
    if data.isna().sum() > 0:
        data = data.fillna(replace_nan)

    return (data - data.min()) / (data.max() - data.min())


def fill_with_ratio(data, remove=None):
    data_count = data.value_counts()
    items = data_count.index.tolist()
    counts = data_count.tolist()

    remove_index = items.index(remove)
    remove_total = counts[remove_index]
    del items[remove_index]
    del counts[remove_index]

    ratio = [x / sum(counts) for x in counts]
    add_per_class = [int(remove_total*i) for i in ratio]
    while sum(add_per_class) != remove_total:
        add_per_class[add_per_class.index(min(add_per_class))] += 1

    for i in range(len(data)):
        if data.loc[i] == remove:
            while True:
                choose_index = random.randint(0, 2)
                if add_per_class[choose_index] != 0:
                    break

            data.loc[i] = items[choose_index]
            add_per_class[choose_index] -= 1

    return data


def process_by_column(data):
    print(data.loc[0])
    # column 0: index, 1: id

    # column index 2, 3
    data = data[data.gender != "Other"]
    data = data.reset_index()

    category, one_hot_df = to_categorical(data["gender"])
    data = data.drop(["gender"], axis=1)
    data.insert(2, "gender_" + category[0], one_hot_df[category[0]])
    data.insert(3, "gender_" + category[1], one_hot_df[category[1]])
    
    # column index 4
    data["age"] = normalization(data["age"])   
    
    # column index 5, 6
    category, one_hot_df = to_categorical(data["hypertension"])
    data = data.drop(["hypertension"], axis=1)
    data.insert(5, "hypertension_" + str(category[0]), one_hot_df[category[0]])
    data.insert(6, "hypertension_" + str(category[1]), one_hot_df[category[1]])

    # column index 7, 8
    category, one_hot_df = to_categorical(data["heart_disease"])
    data = data.drop(["heart_disease"], axis=1)
    data.insert(7, "heart_disease_" + str(category[0]), one_hot_df[category[0]])
    data.insert(8, "heart_disease_" + str(category[1]), one_hot_df[category[1]])

    # column index 9, 10
    category, one_hot_df = to_categorical(data["ever_married"])
    data = data.drop(["ever_married"], axis=1)
    data.insert(9, "ever_married_" + category[0], one_hot_df[category[0]])
    data.insert(10, "ever_married_" + category[1], one_hot_df[category[1]]) 
    
    # column index 11, 12, 13, 14, 15
    category, one_hot_df = to_categorical(data["work_type"])
    data = data.drop(["work_type"], axis=1)
    data.insert(11, "work_type_" + category[0], one_hot_df[category[0]])
    data.insert(12, "work_type_" + category[1], one_hot_df[category[1]])    
    data.insert(13, "work_type_" + category[2], one_hot_df[category[2]])
    data.insert(14, "work_type_" + category[3], one_hot_df[category[3]])
    data.insert(15, "work_type_" + category[4], one_hot_df[category[4]])  
    
    # column index 16, 17
    category, one_hot_df = to_categorical(data["Residence_type"])
    data = data.drop(["Residence_type"], axis=1)
    data.insert(16, "Residence_type_" + category[0], one_hot_df[category[0]])
    data.insert(17, "Residence_type_" + category[1], one_hot_df[category[1]])

    # column index 18
    data["avg_glucose_level"] = normalization(data["avg_glucose_level"])

    # column index 19
    bmi_avg = data["bmi"].sum() / len(data["bmi"])
    data["bmi"] = normalization(data["bmi"], bmi_avg)
    
    # column index 20, 21, 22
    data["smoking_status"] = fill_with_ratio(data["smoking_status"], remove="Unknown")
    category, one_hot_df = to_categorical(data["smoking_status"])
    data = data.drop(["smoking_status"], axis=1)
    data.insert(20, "smoking_status_" + category[0], one_hot_df[category[0]])
    data.insert(21, "smoking_status_" + category[1], one_hot_df[category[1]])
    data.insert(22, "smoking_status_" + category[2], one_hot_df[category[2]])
    print(data.loc[0])
    data = data.drop(["index", "id"], axis=1)

    return data


def preprocess(data_path, save_path):
    data = pd.read_csv(data_path)

    preprocessed_data = process_by_column(data)

    train, test = train_test_split(
        preprocessed_data, 
        test_size=0.2,
        stratify=preprocessed_data[["stroke"]]
    )
    
    columns = list(preprocessed_data.columns)
    x_columns = columns[:-1]
    y_columns = [columns[-1]]

    final_trian_data, final_train_label = SMOTE().fit_resample(
        train[x_columns],
        train[y_columns]
    )
    
    final_test_data = test[x_columns]
    final_test_label = test[y_columns]

    final_trian_data.to_csv(save_path + "train_data.csv", encoding="utf-8", index=False)
    final_train_label.to_csv(save_path + "train_label.csv", encoding="utf-8", index=False)
    final_test_data.to_csv(save_path + "test_data.csv", encoding="utf-8", index=False)
    final_test_label.to_csv(save_path + "test_label.csv", encoding="utf-8", index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="The training data path.")
    parser.add_argument("--save_path", type=str, help="The saving path for preprocessed data.")
    args = parser.parse_args()
    
    preprocess(args.data_path, args.save_path)

    # preprocess(
    #     "data/healthcare-dataset-stroke-data.csv",
    #     "data/"
    # )
