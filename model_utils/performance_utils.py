from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, plot_confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np

class Classification_Score(object):
    """
    Calculate Classification Score Module
    pos_label must be need to calculate precision score, ...
    
    """
    def __init__(self, model, y_true, y_pred, pos_label = 1):
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.pos_label = pos_label

    def performance_score(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, pos_label=self.pos_label)
        recall = recall_score(self.y_true, self.y_pred, pos_label=self.pos_label)
        f1 = f1_score(self.y_true, self.y_pred, pos_label=self.pos_label)
        return [accuracy, precision, recall, f1]

    def print_score(self):
        [accuracy, precision, recall, f1] = self.performance_score()
        print("Accuracy Score: {}".format(accuracy))
        print("Precision Score: {}".format(precision))
        print("Recall Score: {}".format(recall))
        print("F1 Score: {}".format(f1))

    def confusion_matrix(self, X_test):
        plot_confusion_matrix(self.model, X_test, self.y_true) 
        plt.show()

    def plot_roc_curve(self, y_prob):
        fpr, tpr, _ = roc_curve(self.y_true,  y_prob)
        auc = round(roc_auc_score(self.y_true, y_prob),3)
        plt.plot(fpr,tpr,label="AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

        
class Regression_Score(object):
    """
    Calculate Regression Score Module
    
    """
    def __init__(self, model, y_true, y_pred):
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred

    def performance_score(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mape = MAPE(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        return [mse, mae, mape, r2]

    def print_score(self):
        [mse, mae, mape, r2] = self.performance_score()
        print("Mean Squared Error: {}".format(mse))
        print("Mean Absolute Error: {}".format(mae))
        print("Mean Absolute Percentage Error: {}".format(mape))
        print("R squared: {}".format(r2))

def MAPE(y_test, y_pred):
	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 