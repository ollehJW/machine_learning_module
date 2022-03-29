# %%
from statsmodels.api import Logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# %%
class build_model(object):
    """
    Building Classification Model Module
    Possible Model list:
        Logistic Regression
        Decision Tree 
        Random Forest
        Support Vector Machine
        Gaussian Naive Bayes Classifier
        XGBoost

    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def LogisticRegression(self):
        model = Logit(self.y, self.X).fit()
        return model

    def DecisionTree(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(self.X, self.y)
        return model

    def SVM(self, C=1.0, kernel='rbf'):
        model = SVC(C=C, kernel=kernel).fit(self.X, self.y)
        return model

    def RandomForest(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, random_state=None):
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state).fit(self.X, self.y)
        return model
    
    def GaussianNaiveBayes(self):
        model = GaussianNB().fit(self.X, self.y)
        return model

    def XGBoost(self, n_estimators=100, max_depth=None, learning_rate=0.1):
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate).fit(self.X, self.y)
        return model
