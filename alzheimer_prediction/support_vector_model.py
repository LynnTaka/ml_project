from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report


def best_svm(X_test, y_test, X_train, y_train, seed):
    # Create and train svm model
    clf = SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
    clf.fit(X_train, y_train.values.ravel())

    # Predictions
    train_predictions = clf.predict(X_train)
    
    # Evaluate the performance
    train_accuracy = accuracy_score(y_train.values.ravel(), train_predictions)

    test_predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test.values.ravel(), test_predictions)

    print(f'Training: {train_accuracy:.2f}')
    print(f'Testing: {test_accuracy:.2f}')

    # Returning testing predictions for postprocessing purposes
    return test_predictions


def train_svm_multiple(X_test, y_test, X_train, y_train, seed):
    # Set possible hyperparameters
    c = [1, 2, 10, 100]
    degree = [1, 2, 3]
    kernel = ['linear', 'poly', 'rbf']
    decision_function_shape = ['ovo', 'ovr']

    best_accuracy = 0
    best_parameters = None

    for c_val in c:
        for d_val in degree:
            for k_val in kernel:
                for dfs_val in decision_function_shape:
                    # print("Current parameters: c=" + str(c_val) + ", degree=" + str(d_val)
                    #                        + ", kernel=" + k_val + ", decision function shape=" + dfs_val)
                    
                    # Initialize svm classifier
                    clf = svm.SVC(C=c_val, degree=d_val, kernel=k_val, decision_function_shape=dfs_val, random_state=seed)

                    # Fit SVM to training data
                    clf.fit(X_train, y_train.values.ravel())

                    # Predict classes with SVM
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test.values.ravel(), y_pred)
                    # print("Accuracy: "+str(accuracy))

                    # count += 1
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = ("Highest SVM accuracy so far: " + str(best_accuracy) + "\n"
                                           + "Parameters: c=" + str(c_val) + ", degree=" + str(d_val)
                                           + ", kernel=" + k_val + ", decision function shape=" + dfs_val + "\n")

    # Print best parameters outside the loop
    print(best_parameters)
