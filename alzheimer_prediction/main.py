import numpy as np
import tensorflow as tf
from preprocessing import *


if __name__ == '__main__':
    #change path depending on where data is located
    datapath = './alzheimer.csv'

    # preprocessing
    dp = read_data(datapath)
    X, y, new_y = encode_data(dp)
    # find_important_features(X, y)

    # set seed for numpy and tensorflow
    seed = 10
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    new_y = new_y.reset_index(drop=True)

    # split data 70-30
    train_idx = np.random.choice(len(X), round(len(X) * 0.7), replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # print(y_train)
    # print(y_test)



