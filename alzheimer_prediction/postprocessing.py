import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def plot_2D_SVM(x_train, y_train, seed):
    # Train model
    clf = SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
    clf.fit(x_train, y_train.values.ravel())

    # Get weight (coef) of the trained model
    w = clf.coef_[0]

    # Create meshgrid to plot decision boundaries
    xx, yy = np.meshgrid(np.linspace(x_train['CDR'].min()-0.5, x_train['CDR'].max()+0.5, 100),
                        np.linspace(x_train['MMSE'].min()-1, x_train['MMSE'].max()+1, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot data points
    plt.scatter(x_train['CDR'], x_train['MMSE'], c=y_train, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7)

    # Plot decision boundaries and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

    # Plot decision hyperplane
    a = -w[0] / w[1]
    xx_hyperplane = np.linspace(x_train['CDR'].min(), x_train['CDR'].max())
    yy_hyperplane = a * xx_hyperplane - (clf.intercept_[0]) / w[1]

    plt.plot(xx_hyperplane, yy_hyperplane, 'k-')

    # Plot parallel hyperplanes
    b1 = clf.support_vectors_[0]
    yy_top = a * xx_hyperplane + (b1[1] - a * b1[0])
    b2 = clf.support_vectors_[-1]
    yy_bottom = a * xx_hyperplane + (b2[1] - a * b2[0])

    plt.plot(xx_hyperplane, yy_top, 'k--')
    plt.plot(xx_hyperplane, yy_bottom, 'k--')

    # Name title and axis labels
    plt.title('SVM for Dementia Classification')
    plt.xlabel('Clinical Dementia Rating (CDR)')
    plt.ylabel('Mini-Mental State Examination (MMSE)')

    plt.savefig('svm.png')
    plt.show()