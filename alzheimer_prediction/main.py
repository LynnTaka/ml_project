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
    # print("dp:")
    # print(dp)
    X, y, new_y = pre.encode_data(dp)
    # pre.find_important_features(X, y)

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
    
    # Find the best parameters for the SVM; comment when done
    # svmLib.train_svm_multiple(X_train, y_train, X_test, y_test, seed)
    
    ##### UNCOMMENT LATER
    # Train best SVM
    # svmLib.best_svm(X_train, y_train, X_test, y_test, seed)

    # Postprocessing: get the most important features to highlight
    #print("X training:")
    #print(X_train)
    headers = ["M/F", "Age", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    X_train.columns = headers
    #print(X_train)
    # y_train.columns = ["Group"]
    print("y train:")
    print(y_train)
    pre.find_important_features(X_train, y_train)

    # Find best parameters for SVM with top 2 features
    x_reduction = X_train[["CDR", "MMSE"]]
    #x_reduction = X_train.iloc[:, [3, 4]]
    #print("X reduced")
    #print(x_reduction)

    X_test.columns = headers
    x_reduced_test = X_test[["CDR", "MMSE"]]
    # print("x testing reduced")
    # print(x_reduced_test)

    # print("y test")
    # print(y_test)

    #svmLib.train_svm_multiple(x_reduction, y_train, x_reduced_test, y_test, seed)

    # Plot SVM with 2 most important features
    post.create_2D_SVM(x_reduction, y_train, seed)
