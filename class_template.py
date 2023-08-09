"""

Tuning deep learning hyperparameters using Gridsearch


The grid search provided by GridSearchCV exhaustively generates candidates 
from a grid of parameter values specified with the param_grid parameter.


The GridSearchCV instance when “fitting” on a dataset, all the possible 
combinations of parameter values are evaluated and the best combination is retained.

cv parameter can be defined for the cross-validation splitting strategy.

GridSearch is designed to work with models from sklearn. But, we can also use it
to tune deep learning hyper parameters - at least for keras models. 

Wisconsin breast cancer example
Dataset link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


"""

from sklearn.metrics import confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
import seaborn as sns
import pandas as pd
import tensorflow as tf
import sklearn
import sys
("Python version is", sys.version)
print("Scikit-learn version is: ", sklearn.__version__)
print("Tensorflow version is: ", tf.__version__)


df = pd.read_csv("data/wisconsin_breast_cancer_dataset.csv")

print(df.describe().T)  # Values need to be normalized before fitting.
print(df.isnull().sum())
# df = df.dropna()

# Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis': 'Label'})
print(df.dtypes)

# Understand the data
# sns.countplot(x="Label", data=df) #M - malignant   B - benign
df['Label'].value_counts()

# Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values

# Encoding categorical data
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)  # M=1 and B=0
#################################################################
# Define x and normalize values

# Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels=["Label", "id"], axis=1)

# scale only after the splitting train and test
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


####################################################################
# Define the keras model as a function.
# Parameters that need to be tuned can be provided as inputs to the function
# Here, let us tune the number of neurons and the optimizer

# Appropriate architecture for the challenge
def create_model(neurons, optimizer):
    model = Sequential()
    model.add(Dense(neurons, input_dim=30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# define the parameter range
param_grid = {'neurons': [2, 8, 16],
              'batch_size': [4, 16],
              'optimizer': ['SGD', 'RMSprop', 'Adam']}

# 3 x 2 x 3 = 18 combinations for parameters

# Define the model using KerasClassifier method.
# This makes our keras model available for GridSearch
model = KerasClassifier(build_fn=create_model, epochs=10, verbose=1)

# n_jobs=-1 parallelizes but it may crash your system.
# Provide the metric for KFold crossvalidation. cv=3 is a good starting point
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)

# Takes a long time based on the number of parameters and cv splits.
# In our case - 18 * 3 * 2 * num_epochs = 1080 total epochs if epochs=10
grid_result = grid.fit(X, Y)

# summarize results
print("Best accuracy of: %f using %s" % (grid_result.best_score_,
                                         grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

###########################################################

# Let us load the best model and predict on our input data
best_model = grid_result.best_estimator_

# Predicting the Test set results
y_pred = best_model.predict(X)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(Y, y_pred)
sns.heatmap(cm, annot=True)
