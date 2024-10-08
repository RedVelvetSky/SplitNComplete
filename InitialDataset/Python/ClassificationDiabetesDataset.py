import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pprint
import warnings

warnings.filterwarnings('ignore')

# df - data frame
df = pd.read_csv("E:\Stuff\introai\\09-decision_tree_diabetes\diabetes_changed.csv")

"""
So, for example, if you look at the row for "Glucose":
The number in the first column (99.00000) means that 25% of the people have glucose levels lower than 99.
The number in the second column (117.0000) is the median, which means half of the people have glucose levels lower than 117 and half have higher.
The number in the third column (140.25000) means that 75% of the people have glucose levels lower than 140.25.
"""

quantiles = df.iloc[:, :-1].quantile(q=[0.2, 0.5, 0.8], axis=0, numeric_only=True).T
diabetes_distribution = df['Outcome'].value_counts() * 100 / len(df)
print(f"\nPercentage of people with features lower than:\n{quantiles}")
print(f"\nDistribution of diabetes presence (in %):\n{diabetes_distribution}")

columns_cant_have_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_cant_have_zeros] = df[columns_cant_have_zeros].replace(0, np.NaN)
print('\n', df[columns_cant_have_zeros])
print('\n Number of NaN values', df.isnull().sum())

# Remove rows with missing values
cleaned_data = df.dropna()

# Group the cleaned data by the 'Outcome' column
outcome_groups = cleaned_data.groupby('Outcome')

# Reset the index within each group to have consecutive row numbers
grouped_data = outcome_groups.apply(lambda group: group.reset_index(drop=True))

# Compute the median for those with and without diabetes
median_diabetes = grouped_data.loc[grouped_data['Outcome'] == 1].median()
median_no_diabetes = grouped_data.loc[grouped_data['Outcome'] == 0].median()

# Print the medians
print("\nMedian for individuals with diabetes:")
print(median_diabetes)

print("\nMedian for individuals without diabetes:")
print(median_no_diabetes)

for column in columns_cant_have_zeros:
    df.loc[(df['Outcome'] == 0) & (df[column].isnull()), column] = median_no_diabetes[column]
    df.loc[(df['Outcome'] == 1) & (df[column].isnull()), column] = median_diabetes[column]

print('\nChecking number of zero values', df.isnull().sum())
# separating output and features
y = df['Outcome']
x = df.iloc[:, :-1]

# dividing data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# building model
model = DecisionTreeClassifier()
kfold = KFold(n_splits=10)

'''
Cross-validation is a technique used in machine learning to assess the performance of a predictive model. It helps to evaluate how well the model will generalize to new, unseen data. The basic idea behind cross-validation is to split the dataset into multiple subsets, train the model on some of these subsets, and then evaluate its performance on the remaining subsets.
'''

# estimating accuracy
cv_results = cross_val_score(model, x, y, cv = 10, scoring='accuracy')

# Classification and Regression Trees (CART)
print(f"\nDecision tree using cross-val: \n\tMean accuracy: {round(cv_results.mean(), 5)} \n\tStd: {round(cv_results.std(), 5)}")

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train,Y_train)
y_predicted = decision_tree.predict(X_test)

# Classification and Regression Trees (CART)
print(f"\nDecision tree no cross-val: \n\tAccuracy: {round(accuracy_score(Y_test, y_predicted), 5)}" f"\n\tPrecision: {round(precision_score(Y_test, y_predicted), 5)}" f"\n\tRecall: {round(recall_score(Y_test, y_predicted), 5)}")

# model tuning
decision_tree = DecisionTreeClassifier(random_state=42)

# Define the parameter grid based on the parameter options you provided
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 6, 7, 8, 9, 10, ],
    'min_samples_split': [1, 2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'ccp_alpha': [0, .001, .005, .01, .05, .1]
}

print('\n')
# GridSearchCV with the decision tree and the parameter grid
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# fit GridSearchCV to the training data
grid_search.fit(X_train, Y_train)

# best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

# using the best estimator to make predictions
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)

print("Accuracy on test set: {:.5f}".format(accuracy_score(Y_test, y_pred)))
print("Precision on test set: {:.5f}".format(precision_score(Y_test, y_pred)))
print("Recall on test set: {:.5f}".format(recall_score(Y_test, y_pred)))

best_params = grid_search.best_params_
model = DecisionTreeClassifier(**best_params)

train_sizes = np.linspace(0.1, 0.9, 9)  # Train sizes from 10% to 90%
accuracies = []
precisions = []
recalls = []

# Iterate over different train sizes
for train_size in train_sizes:
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(Y_test, Y_pred))
    precisions.append(precision_score(Y_test, Y_pred))
    recalls.append(recall_score(Y_test, Y_pred))

# Plotting Accuracy, Precision, and Recall vs. Train-Test Proportion
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, accuracies, label='Accuracy')
plt.plot(train_sizes, precisions, label='Precision')
plt.plot(train_sizes, recalls, label='Recall')
plt.xlabel('Proportion of Training Data')
plt.ylabel('Performance Metrics')
plt.title('Model Performance vs. Train-Test Split Proportion')
plt.legend()
plt.grid(True)
plt.show()

# Visualizing the Decision Tree using the last trained model
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=df.columns[:-1], class_names=["No Diabetes", "Diabetes"])
plt.title('Decision Tree Visualization')
plt.show()