import copy
import numpy as np


class RandomForest:
    '''Random Forest Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]

    def _get_bootstrap_dataset(self, X, y):
        """Create a bootstrap dataset for X.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).

        Returns:
            X_bootstrap: a sampled dataset, of shape (N, D).
            y_bootstrap: the labels for sampled dataset.
        """
        # YOUR CODE HERE
        # TODO: re‐sample N examples from X with replacement
        # begin answer
        import random
        col=[]
        while len(col)<X.shape[0]:
            _idx=random.randint(0,X.shape[0]-1)
            col.append(_idx)
        X_bootstrap=X.iloc[col,]
        y_bootstrap=y.iloc[col,]
        return X_bootstrap,y_bootstrap
        # end answer

    def fit(self, X, y):
        """Build the random forest according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        for base_model in self._estimators:
            X_bootstrap,y_bootstrap=self._get_bootstrap_dataset(X, y)
            base_model.fit(X_bootstrap,y_bootstrap)
        # end answer
        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        # YOUR CODE HERE
        # begin answer
        for i in range(N):
            _y=[int(base_model.predict(X.iloc[[i]])) for base_model in self._estimators]
            y_pred[i]=max(set(_y),key=_y.count)
        # end answer
        return y_pred
