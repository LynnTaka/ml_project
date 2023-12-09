import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets, svm
import seaborn as sns

def plot_SVM(x_train, y_train, seed):
    clf = SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
    clf.fit(x_train, y_train.values.ravel())

    # Get the weight (coef) and bias (intercept) from the trained model
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Create a meshgrid to plot decision boundaries
    xx, yy = np.meshgrid(np.linspace(x_train['CDR'].min()-0.5, x_train['CDR'].max()+0.5, 100),
                        np.linspace(x_train['MMSE'].min()-1, x_train['MMSE'].max()+1, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the data points
    plt.scatter(x_train['CDR'], x_train['MMSE'], c=y_train, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7)

    # Plot the decision boundaries and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

    # Plot the hyperplane
    a = -w[0] / w[1]
    xx_hyperplane = np.linspace(x_train['CDR'].min(), x_train['CDR'].max())
    yy_hyperplane = a * xx_hyperplane - (clf.intercept_[0]) / w[1]

    plt.plot(xx_hyperplane, yy_hyperplane, 'k-')

    # Plot the parallel hyperplanes
    b1 = clf.support_vectors_[0]
    yy_top = a * xx_hyperplane + (b1[1] - a * b1[0])
    b2 = clf.support_vectors_[-1]
    yy_bottom = a * xx_hyperplane + (b2[1] - a * b2[0])

    plt.plot(xx_hyperplane, yy_top, 'k--')
    plt.plot(xx_hyperplane, yy_bottom, 'k--')

    # Set labels and show the plot
    plt.title('SVM')
    plt.xlabel('CDR')
    plt.ylabel('MMSE')

    plt.savefig('svm3.png')
    plt.show()

def create_2D_SVM(x_train, y_train, seed):
    clf = SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
    clf.fit(x_train, y_train.values.ravel())

    # Create a meshgrid for the plot
    xx, yy = np.meshgrid(np.linspace(x_train['CDR'].min()-1, x_train['CDR'].max()+1, 100),
                        np.linspace(x_train['MMSE'].min()-1, x_train['MMSE'].max()+1, 100))

    # Plot decision boundary
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linewidths=2, colors='black')

    # Plot data points
    plt.scatter(x_train['CDR'], x_train['MMSE'], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o')

    # Highlight support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')

    # Set labels and show the plot
    plt.title('SVM')
    plt.xlabel('CDR')
    plt.ylabel('MMSE')
    
    plt.savefig('svm.png')
    plt.show()



# def create_2D_SVM(x_train, y_train, seed):
#     clf = SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
#     clf.fit(x_train, y_train.values.ravel())

#     # Plot SVM
#     # Create 2-feature plot
#     plt.figure(figsize=(10, 8))
#     # print("X train")
#     # print(type(x_train))
#     # print(x_train)
#     # print("Y train")
#     # print(type(y_train))
#     # print(y_train)

#     # x1_temp = x_train['CDR']
#     # print("X1 temp")
#     # print(type(x1_temp))
#     # print(x1_temp)
#     # x2_temp = x_train['MMSE']
#     # print("X2 temp")
#     # print(type(x2_temp))
#     # print(x2_temp)
#     # y_temp = y_train.values.ravel()
#     # print("y temp")
#     # print(type(y_temp))
#     # print(y_temp)

#     # might need to change y_train
#     sns.scatterplot(x=x_train['CDR'], y=x_train['MMSE'], hue=y_train.values.ravel(), s=8)
#     # sns.scatterplot(x=x_train.iloc[0].ravel(), y=x_train.iloc[1].ravel(), hue=y_train.values.ravel(), s=8)

#     # Create hyperplane
#     w = clf.coef_[0]
#     print("w")
#     print(w)
#     b = clf.intercept_[0]
#     x_points = np.linspace(-1, 1)
#     y_points = -(w[0]/w[1]) * x_points - b / w[1]

#     # Plot red hyperplant
#     plt.plot(x_points, y_points, c='r')

    


# def create_2D_SVM(x, y, seed):
#     plt.figure(figsize=(10, 8))

#     # Constructing a hyperplane using a formula.
#     w = svc_model.coef_[0]           # w consists of 2 elements
#     b = svc_model.intercept_[0]      # b consists of 1 element
#     x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
#     y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
#     # Plotting a red hyperplane
#     plt.plot(x_points, y_points, c='r')



    #plt.scatter(x.iloc[0], x.iloc[1], c=y, s=50, cmap='autumn')
    #plt.show()
    # clf = svm.SVC(C=1, degree=1, kernel='linear', decision_function_shape='ovo', random_state=seed)
    # clf.fit(x, y.values.ravel())

    # print("Fitted clf")

    # axes = plt.gca()
    # xLimits = axes.get_xlim()
    # yLimits = axes.get_ylim()

    # xx, yy = np.meshgrid(np.linspace(xLimits[0], xLimits[1], len(x)), np.linspace(yLimits[0], yLimits[1], len(y)))
    # z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # z = z.reshape(xx.shape)
    # plt.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

    # #c=y  ->  c=y.values.ravel()
    # plt.scatter(x.iloc[0], x.iloc[1], c=y.values.ravel(), cmap=plt.cm.Paired, edgecolors='k', marker='o')

    # plt.title('SVM Decision Boundary with Two Features')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()