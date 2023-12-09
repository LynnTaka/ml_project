import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# gets data from csv file and returns data frame obj
def read_data(data_path):
    # Read CSV
    df = pd.read_csv(data_path, sep=',', skiprows=1, header=None)

    # check to see if there are missing values in any row
    # print("EMPTY FIELDS:")
    # print(df[pd.isnull(df).any(axis=1)])
    # print(df.isna().sum())

    # drop the column SES, drop education
    df = df.drop(df.columns[4], axis=1)
    df = df.drop(df.columns[3], axis=1)

    # drop rows with missing data
    df.dropna(inplace=True)

    return df


# transforms and cleans dataset for use
def encode_data(df):
    print("encode_data")

    # encode categorical variables
    label_encoder = LabelEncoder()
    df[1] = label_encoder.fit_transform(df[1])

    # split into inputs and outputs
    X = df.drop(df.columns[0], axis=1)
    y = df[0]

    # Encode "Demented" and "Converted" as 1 and "Nondemented" as 0
    temp_dict = {'Nondemented': 0, 'Demented': 1, 'Converted': 1}

    # encode and create a new df
    new_y = y.map(temp_dict).to_frame()

    return X, y, new_y


# print out which features are most important using rf
def find_important_features(X, y):
    # print("find_important_features")

    # initialize and fit model to data
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # get feature importance from pre-built model
    important_features = rf.feature_importances_

    # display important features
    important_df = pd.DataFrame({'Feature': X.columns, 'Importance': important_features})
    important_df = important_df.sort_values(by='Importance', ascending=False)
    print(important_df)