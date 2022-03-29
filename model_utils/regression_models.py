# %%
from statsmodels.api import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# %%
class build_model(object):
    """
    Building Regression Model Module
    Possible Model list:
        Linear Regression:
            Mode: Ridge, Lasso, ElasticNet
        Decision Tree 
        Random Forest
        Support Vector Machine
        XGBoost
 
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def LinearRegression(self, alpha=0.01, mode=None):
        if mode==None:
            model = OLS(self.y, self.X).fit()
        elif mode=='Ridge':
            model = OLS(self.y, self.X).fit_regularized(alpha=alpha, L1_wt=0)
        elif mode=='Lasso':
            model = OLS(self.y, self.X).fit_regularized(alpha=alpha, L1_wt=1)
        elif mode=='ElasticNet':
            model = OLS(self.y, self.X).fit_regularized(alpha=alpha, L1_wt=0.5)
        return model

    def DecisionTree(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(self.X, self.y)
        return model

    def SVM(self, C=1.0, kernel='rbf'):
        model = SVR(C=C, kernel=kernel).fit(self.X, self.y)
        return model

    def RandomForest(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state).fit(self.X, self.y)
        return model

    def XGBoost(self, n_estimators=100, max_depth=None, learning_rate=0.1):
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate).fit(self.X, self.y)
        return model
