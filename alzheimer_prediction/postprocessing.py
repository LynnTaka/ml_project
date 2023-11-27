# from sklearn.svm import SVC
# from sklearn import svm

# def umap_visualization(training_predictions):

import umap
from sklearn.model_selection import GridSearchCV

def umap_GridSearch(feature_set, classification):
    param_grid = {
        'n_neighbors': [5, 10, 15, 20],
        'min_dist': [0.1, 0.5, 0.8],
        'n_components': [2, 3]
    }

    umap_model = umap.UMAP()
    print("UMAP done")
    grid_search = GridSearchCV(umap_model, param_grid, cv=5, scoring='accuracy')  # Use an appropriate scoring metric
    print("GS called")
    grid_search.fit(feature_set, classification)
    print("GS fitted")

    # Extract best parameters and best UMAP model
    #best_params = grid_search.best_params_
    #best_umap_model = grid_search.best_estimator_

    # Print the best parameters
    #print("Best Parameters:", best_params)

    # Access other information from grid_search, such as the best score
    #print("Best Score:", grid_search.best_score_)

