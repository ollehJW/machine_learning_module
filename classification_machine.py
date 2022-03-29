# %% Import Modules
from preprocess_utils.data_utils import load_data, one_hot_encoder, NA_Cheak
from preprocess_utils.normalization import Normaize
from preprocess_utils.class_imbalance import balancing_category
from model_utils.performance_utils import Classification_Score
from model_utils.classification_models import build_model
from collections import Counter
from sklearn.model_selection import train_test_split

# %% Load data
data = load_data('/home/jongwook95.lee/study/machine_learning/train.csv')
data.head()
# %% Count NA per columns
NA_Cheaker = NA_Cheak(data)
NA_Cheaker.NA_count()
# %% Handle NAs
NA_Cheaker.handling_NA(columns = ['Age'], method = 'fill_mean')
NA_Cheaker.handling_NA(columns = ['Embarked'], method = 'fill_value', fill_value='S')
NA_Cheaker.NA_count()
# %% One hot encoding
numeric_variable = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_variable = ['Pclass', 'Sex', 'Embarked']
target_variable = 'Survived'
dummied_data = one_hot_encoder(data, numeric_variable, categorical_variable, target_variable)
dummied_data.head()
# %% Normalize
Normaize(dummied_data, numeric_variable, method = 'min-max')
dummied_data.head()
# %% Class Imbalance Problem Handling
Cb_handling = balancing_category(X = dummied_data.iloc[:, :-1], y = dummied_data[target_variable])
Cb_handling.class_count()
sampled_X, sampled_y = Cb_handling.SMOTE(k_neighbors = 3)
print('Resampling is conducted.')
Counter(sampled_y)
# %% Train Test split
test_split = 0.3
X_train, X_test, y_train, y_test = train_test_split(sampled_X, sampled_y,
                                                    stratify=sampled_y, 
                                                    test_size=test_split)
# %% Modeling
Models = build_model(X_train, y_train)
# %% (1) Logistic Regression
lr_model = Models.LogisticRegression()
print(lr_model.summary())
y_prob = lr_model.predict(X_test)
y_pred = list(map(round, y_prob))
lr_model_scoring = Classification_Score(lr_model, y_test, y_pred)
lr_model_scoring.print_score()
# lr_model_scoring.confusion_matrix(X_test)
lr_model_scoring.plot_roc_curve(y_prob)
# %% (2) Decision Tree
dt_model = Models.DecisionTree()
y_pred = dt_model.predict(X_test)
dt_model_scoring = Classification_Score(dt_model, y_test, y_pred)
dt_model_scoring.print_score()
dt_model_scoring.confusion_matrix(X_test)
dt_model_scoring.plot_roc_curve(dt_model.predict_proba(X_test)[:,1])
# %% (3) SVM
svm_model = Models.SVM()
y_pred = svm_model.predict(X_test)
svm_model_scoring = Classification_Score(svm_model, y_test, y_pred)
svm_model_scoring.print_score()
svm_model_scoring.confusion_matrix(X_test)
# svm_model_scoring.plot_roc_curve(svm_model.predict_proba(X_test, probability=)[:,1])
# %% Randomforest
rf_model = Models.RandomForest()
y_pred = rf_model.predict(X_test)
rf_model_scoring = Classification_Score(rf_model, y_test, y_pred)
rf_model_scoring.print_score()
rf_model_scoring.confusion_matrix(X_test)
rf_model_scoring.plot_roc_curve(rf_model.predict_proba(X_test)[:,1])
# %% Gaussian Naive Bayes
gnb_model = Models.GaussianNaiveBayes()
y_pred = rf_model.predict(X_test)
gnb_model_scoring = Classification_Score(gnb_model, y_test, y_pred)
gnb_model_scoring.print_score()
gnb_model_scoring.confusion_matrix(X_test)
gnb_model_scoring.plot_roc_curve(gnb_model.predict_proba(X_test)[:,1])
# %% XGBoost
xgb_model = Models.XGBoost()
y_pred = rf_model.predict(X_test)
xgb_model_scoring = Classification_Score(xgb_model, y_test, y_pred)
xgb_model_scoring.print_score()
xgb_model_scoring.confusion_matrix(X_test)
xgb_model_scoring.plot_roc_curve(xgb_model.predict_proba(X_test)[:,1])
# %%
