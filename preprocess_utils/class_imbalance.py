from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pandas as pd

class balancing_category(object):
    """
    Class Imbalance Problem Handling Module 
    method: (Undersampling, Oversampling, SMOTE)
    Your input should contain four information:
        X: Input Variables
        y: Target variable (Class Imbalance) 
        Method: Handling Methods:
            Undersampling, Oversampling, SMOTE

    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def class_count(self):
        print(Counter(self.y))
    
    def under_sampling(self):
        rus = RandomUnderSampler()
        undersampled_data, undersampled_label = rus.fit_resample(self.X, self.y)
        undersampled_data = pd.DataFrame(undersampled_data, columns=self.X.columns)
        return undersampled_data, undersampled_label
    
    def over_sampling(self):
        ros = RandomOverSampler()
        oversampled_data, oversampled_label = ros.fit_resample(self.X, self.y)
        oversampled_data = pd.DataFrame(oversampled_data, columns=self.X.columns)
        return oversampled_data, oversampled_label

    def SMOTE(self, k_neighbors = 5):
        smote = SMOTE(k_neighbors = k_neighbors)
        smoted_data, smoted_label = smote.fit_resample(self.X, self.y)
        return smoted_data, smoted_label