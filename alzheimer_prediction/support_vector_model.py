from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report


def train_svm(X_test, y_test, X_train, y_train, seed):
    # create and train svm model
    clf = SVC(kernel='linear', random_state=seed)
    clf.fit(X_train, y_train.values.ravel())

    # predictions
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    # Evaluate the performance
    train_accuracy = accuracy_score(y_train.values.ravel(), train_predictions)
    test_accuracy = accuracy_score(y_test.values.ravel(), test_predictions)

    print(f'Training: {train_accuracy:.2f}')
    print(f'Testing: {test_accuracy:.2f}')


def train_svm_multiple(X_test, y_test, X_train, y_train, seed):
    print('inside train svm multiple')

    # define hyperparameters
    c = [1, 2, 10, 100]
    degree = [1, 2, 3]
    kernel = ['linear', 'poly', 'rbf']
    decision_function_shape = ['ovo', 'ovr']

    best_accuracy = 0
    best_parameters = None

    print(type(X_train))
    print(type(y_train))
    print(type(X_test))
    print(type(y_test))

    for c_val in c:
        for d_val in degree:
            for k_val in kernel:
                for dfs_val in decision_function_shape:
                    # initialize svm classifier
                    clf = svm.SVC(C=c_val, degree=d_val, kernel=k_val, decision_function_shape=dfs_val)

                    # Fit SVM to training data
                    clf.fit(X_train, y_train)

                    accuracy = 0

                    # for x_test_sample, y_test_sample in zip(X_test, y_test):
                    #     prediction = clf.predict(x_test_sample.reshape(1,-1))
                    #     if prediction == y_test_sample:
                    #         accuracy += 1

                    for x_test_sample, y_test_sample in zip(X_test, y_test):
                        # predictions
                        prediction = clf.predict([x_test_sample].reshape(1,-1))
                        if prediction == y_test_sample:
                            accuracy += 1


                    # Calculate overall accuracy
                    accuracy /= len(y_test)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameters = ("Highest SVM accuracy so far: " + str(best_accuracy) + "\n"
                                           + "Parameters: c=" + str(c_val) + ", degree=" + str(d_val)
                                           + ", kernel=" + k_val + ", decision function shape=" + dfs_val + "\n")

        # Print best parameters outside the loop
    print(best_parameters)
