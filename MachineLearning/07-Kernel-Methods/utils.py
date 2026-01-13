import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import PolynomialFeatures


class AbcClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # Store the classes seen during fit
        if y is not None:
            self.classes_ = unique_labels(y)
            X, y = check_X_y(X, y)
            out = X, y
        else:
            X = check_array(X)
            out = X
        return out

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X


class Dataset():
    def __init__(self, mode):
        self.mode = mode

    def make(self, n_samples=1500, n_features=2, n_classes=2, noise=0.3, seed=1):
        if self.mode == 'linear':
            X, y = datasets.make_classification(n_samples=n_samples,
                                                n_features=n_features, 
                                                n_classes=n_classes,
                                                n_informative=1,
                                                n_redundant=0, 
                                                n_clusters_per_class=1, 
                                                random_state=seed)
            rng = np.random.RandomState(seed)
            X += noise * rng.uniform(size=X.shape)

        elif self.mode == 'moon':
            X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        elif self.mode == 'circle':
            X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.2, random_state=seed)
        elif self.mode == 'blob':
            pass

        y = 2 * y - 1 

        return X, y


class VisDecision():
    def __init__(self, clf, plot_method='contourf', response_method='predict'):
        """
        plot_method: {'contourf', 'contour', 'pcolormesh'}, default='contourf'}
        response_method: {'auto', 'predict_proba', 'decision_function', 'predict'}, 
        default='auto'}
        """
        self.clf = clf
        self.plot_method = plot_method
        self.response_method = response_method
        self.cm = plt.cm.RdBu
        self.cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    def plot(self, X, y):
        assert X.shape[-1] == 2
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)

        DecisionBoundaryDisplay.from_estimator(
            self.clf, X, cmap=self.cm, alpha=0.8, ax=ax, eps=0.5,
            xlabel="X1", ylabel="X2",plot_method=self.plot_method,
            response_method = self.response_method
        )

        ax.scatter(X[:, 0], X[:, 1], c=y, s=80, cmap=self.cm_bright, edgecolors="k")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
#         ax.set_xticks(())
#         ax.set_yticks(())

        plt.title("Decision Boundary")

        return ax
