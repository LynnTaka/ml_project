from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_svm(X_test, y_test, X_train, y_train, seed):
    # create and train svm model
    clf = SVC(kernel='linear',random_state=seed)
    clf.fit(X_train, y_train.values.ravel())

    # predictions
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    # Evaluate the performance
    train_accuracy = accuracy_score(y_train.values.ravel(), train_predictions)
    test_accuracy = accuracy_score(y_test.values.ravel(), test_predictions)

    print(f'Training: {train_accuracy:.2f}')
    print(f'Testing: {test_accuracy:.2f}')
