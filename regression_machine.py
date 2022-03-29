# %% Import Modules
from preprocess_utils.data_utils import load_data, one_hot_encoder, NA_Cheak
from preprocess_utils.normalization import Normaize
from model_utils.performance_utils import Regression_Score
from model_utils.regression_models import build_model
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
numeric_variable = ['SibSp', 'Parch', 'Fare']
categorical_variable = ['Pclass', 'Sex', 'Embarked']
target_variable = 'Age'
dummied_data = one_hot_encoder(data, numeric_variable, categorical_variable, target_variable)
dummied_data.head()
# %% Normalize
Normaize(dummied_data, numeric_variable, method = 'min-max')
dummied_data.head()
# %% Train Test split
test_split = 0.3
X_train, X_test, y_train, y_test = train_test_split(dummied_data.iloc[:, :-1], dummied_data[target_variable],
                                                    test_size=test_split)

# %%
Models = build_model(X_train, y_train)
# %% (1) Linear Regression
lr_model = Models.LinearRegression()
print(lr_model.summary())
y_pred = lr_model.predict(X_test)
lr_model_scoring = Regression_Score(lr_model, y_test, y_pred)
lr_model_scoring.print_score()
# %% (2) ElasticNet Regression
lr_model = Models.LinearRegression(alpha=0.01, mode='ElasticNet')
#print(lr_model.summary())
y_pred = lr_model.predict(X_test)
lr_model_scoring = Regression_Score(lr_model, y_test, y_pred)
lr_model_scoring.print_score()

# %% (3) Decision Tree
dt_model = Models.DecisionTree()
y_pred = dt_model.predict(X_test)
dt_model_scoring = Regression_Score(dt_model, y_test, y_pred)
dt_model_scoring.print_score()

# %% (4) SVM
svm_model = Models.SVM()
y_pred = svm_model.predict(X_test)
svm_model_scoring = Regression_Score(svm_model, y_test, y_pred)
svm_model_scoring.print_score()

# %% (5) RandomForest
rf_model = Models.RandomForest()
y_pred = rf_model.predict(X_test)
rf_model_scoring = Regression_Score(rf_model, y_test, y_pred)
rf_model_scoring.print_score()

# %% (6) XGBoost
xgb_model = Models.XGBoost()
y_pred = xgb_model.predict(X_test)
xgb_model_scoring = Regression_Score(xgb_model, y_test, y_pred)
xgb_model_scoring.print_score()
# %%
