import numpy as np
import tensorflow as tf
import preprocessing as pre
import postprocessing as post
import support_vector_model as svmLib
# from preprocessing import *
# from postprocessing import * 
# from support_vector_model import *

if __name__ == '__main__':
    #change path depending on where data is located
    datapath = './alzheimer.csv'

    # preprocessing
    dp = pre.read_data(datapath)
    X, y, new_y = pre.encode_data(dp)
    # find_important_features(X, y)

    # set seed for numpy and tensorflow
    seed = 10
    np.random.seed(seed)

    # reset index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_new = new_y.reset_index(drop=True)

    # split data 70-30
    train_idx = np.random.choice(len(X), round(len(X) * 0.7), replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    X_train = X.loc[train_idx]
    y_train = y_new.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y_new.loc[test_idx]
    print(type(X_train))
    print(type(y_train))
    print(type(X_test))
    print(type(y_test))

    # print(y_train)
    # print(y_test)

    svmLib.train_svm_multiple(X_train, y_train, X_test, y_test, seed)



