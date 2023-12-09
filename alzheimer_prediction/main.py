import numpy as np
import preprocessing as pre
import postprocessing as post
import support_vector_model as svmLib

if __name__ == '__main__':
    # Set path to location of data
    datapath = './alzheimer.csv'

    # PREPROCESSING
    dp = pre.read_data(datapath)
    X, y, new_y = pre.encode_data(dp)
    # pre.find_important_features(X, y)

    # Set seed for numpy
    seed = 10
    np.random.seed(seed)

    # Reset index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_new = new_y.reset_index(drop=True)

    # Split data 70-30
    train_idx = np.random.choice(len(X), round(len(X) * 0.7), replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    X_train = X.loc[train_idx]
    y_train = y_new.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y_new.loc[test_idx]
    
    # Find the best parameters for the SVM; comment when done
    # svmLib.train_svm_multiple(X_train, y_train, X_test, y_test, seed)
    
    # Train best SVM
    svmLib.best_svm(X_train, y_train, X_test, y_test, seed)

    # POSTPROCESSING
    # Find 2 most important features for lower-dimension SVM
    headers = ["M/F", "Age", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    X_train.columns = headers
    # Uncomment if you want to see important features
    # pre.find_important_features(X_train, y_train)

    # Find best parameters for SVM with top 2 features
    x_train_reduced = X_train[["CDR", "MMSE"]]
    # X_test.columns = headers
    # x_test_reduced = X_test[["CDR", "MMSE"]]
    
    # Uncomment to test parameters to find best SVM for reduced feature set
    # svmLib.train_svm_multiple(x_train_reduced, y_train, x_test_reduced, y_test, seed)

    # Plot SVM with 2 most important features
    post.plot_2D_SVM(x_train_reduced, y_train, seed)