import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# gets data from csv file and returns data frame obj
def read_data(data_path):
    # put csv into object
    df = pd.read_csv(data_path)

    # check the head of data
    # print(df.head())

    # 373 rows in dataset
    # print(df.shape)

    # check to see if there are missing values in any row
    # print("EMPTY FIELDS:")
    # print(df[pd.isnull(df).any(axis=1)])
    # print(df.isna().sum())

    # drop the rows where the value is converted instead of dementia and nondem
    # df = df[df['Group'] != 'Converted']

    # drop the column SES
    df = df.drop(['SES'], axis=1)
    # print(df.shape)


    # drop rows with missing data
    df.dropna(inplace=True)

    return df


# transforms and cleans dataset for use
def encode_data(df):
    print("encode_data")

    # encode categorical variables
    label_encoder = LabelEncoder()
    df['M/F'] = label_encoder.fit_transform(df['M/F'])

    # split into inputs and outputs
    X = df.drop('Group', axis=1)
    y = df['Group']

    # encode as everything that is demented as 1 and nondemented as 0
    temp_dict = {'Nondemented': 0, 'Demented': 1, 'Converted': 1}

    # encode and create a new df
    new_y = y.map(temp_dict).rename('Encoded_Group').to_frame()

    # print(new_y)


    return X, y, new_y

# print out which features are most important using rf
def find_important_features(X, y):
    # print("find_important_features")
    
    # initialize and fit model to data
    rf = RandomForestClassifier()
    rf.fit(X,y)

    # get feature importance from pre-built model
    important_features = rf.feature_importances_

    # display important features 
    important_df = pd.DataFrame({'Feature': X.columns, 'Importance': important_features})
    important_df = important_df.sort_values(by='Importance', ascending=False)
    print(important_df)



