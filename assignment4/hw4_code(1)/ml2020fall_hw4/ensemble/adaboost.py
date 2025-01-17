import copy
import numpy as np


class Adaboost:
    '''Adaboost Classifier.

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
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        weight=np.ones(X.shape[0])/X.shape[0]
        #sequential
        for idx in range(self.n_estimator):
            self._estimators[idx].fit(X,y,sample_weights=weight)
            y_pre=self._estimators[idx].predict(X)
            error=np.sum(weight[y_pre!=y])/np.sum(weight)  #compute error
            self._alphas[idx]=np.log((1-error)/error)      #log odds
            weight*=np.exp(-0.5*self._alphas[idx]*y_pre*y) #update weight
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
        for i in range(self.n_estimator):
            y_pred+=self._alphas[i]*self._estimators[i].predict(X)
        y_pred=np.sign(y_pred)
        # end answer
        return y_pred
