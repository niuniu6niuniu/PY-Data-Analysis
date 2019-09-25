# Data analysis and Wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# # # # #     Acquire Data     # # # # #
train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")
combine = [train_df, test_df]
# Features of the dataset
# print(train_df.columns.values)
# - Categorical: Survived, Sex, Embarked
# - Ordinal: Pclass
# - Continuous: Age, Fare
# - Discrete: SibSp, Parch
# =============================================================
# Preview the data
# train_df.head()
# train_df.tail()

# See the data type
# train_df.info()
# print('_' * 40)
# test_df.info()

# Show the statistics
# train_df.describe()
# train_df.describe(include=['0'])
# =============================================================


# # # # #     Analyzing pivot features     # # # # #
# =============================================================
pclass = train_df[['Pclass', 'Survived']].groupby(['Pclass'], \
         as_index=False).mean().sort_values(by='Survived', ascending=False)
sex = train_df[['Sex', 'Survived']].groupby(['Sex'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)
sibSp = train_df[['SibSp', 'Survived']].groupby(['SibSp'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)
parch = train_df[['Parch', 'Survived']].groupby(['Parch'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)
# =============================================================


# # # # #     Analyze by visualizing data     # # # # #

# Correlating numerical features - Observations:
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# =============================================================
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
# =============================================================

# Correlating numerical and ordinal feature - Observations:
# - Pclass=3 had most passengers, however most did not survive.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived.
# - Most passengers in Pclass=1 survived.
# - Pclass varies in terms of Age distribution of passengers.
# =============================================================
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()
# =============================================================

# Correlating categorical features
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.
# =============================================================
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# plt.show()
# =============================================================

# Correlating categorical and numerical features
# - Consider banding Fare feature.
# =============================================================
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()
# =============================================================

# # # # #     Wrangle data     # # # # #

# - Dropping features: Cabin, Ticket
# =============================================================
# print("Before: ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)
combine = [train_df, test_df]
# print("After: ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# =============================================================

# - Creating new feature
# - Extract Title from Name using regular expression
# - Replace titles with a more common name or classify as Rare
#  Title visualization
# =============================================================
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# tab1 = pd.crosstab(train_df['Title'], train_df['Sex'])
# print(tab1)
# =============================================================

# Replace titles
# =============================================================
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
                        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
tab2 = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# print(tab2)
# =============================================================
# - Convert categorical titles to ordinal
# =============================================================
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
# print(train_df.head())
# =============================================================

# - Drop the Name feature in training set and testing set
# - Drop PassengerId feature in training set
# =============================================================
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
# print(train_df.shape, test_df.shape)
# =============================================================

# # #   Convert categorical feature
# - Convert Sex: Gender(male: 0, female: 1)
# =============================================================
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# print(train_df.head())
# =============================================================

# - Completing a numerical continuous feature
# - Age: Guess missing values using other correlated features.
#  In our case we note correlation among Age, Gender, and Pclass.
#  Guess Age values using median values for Age across sets of
#  Pclass and Gender feature combinations.
# =============================================================
# grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()
# =============================================================

# Prepare an empty array to contain guessed Age values based on Pclass x Gender
# Iterate over Sex(0 or 1) and Pclass(1, 2, 3) to calculate guessed ages
# =============================================================
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess / 0.5 + 0.5 ) * 0.5

    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & \
                         (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
            dataset.replace(np.nan, 0, inplace=True)
            dataset.replace(np.inf, 0, inplace=True)
            dataset['Age'] = dataset['Age'].astype(int)
# print(train_df.head())
# =============================================================

# - Create Age bands
# =============================================================
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
tab3 = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], \
        as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# - Replace Age with ordinals based on these bands
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age' ] = 0
    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# print(train_df.head())
# Remove AgeBand feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
# print(train_df.head())
# =============================================================

# - Create new feature for FamilySize which combines Parch and SibSp
# =============================================================
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
tab4 = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], \
                as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(tab4)

# - Create another feature: IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
tab5 = train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# print(tab5)

# - Drop Parch, SibSp and FamilySize
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
# print(train_df.head())
# =============================================================

# - We can also create an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age * Class'] = dataset.Age * dataset.Pclass
tab6 = train_df.loc[:, ['Age * Class', 'Age', 'Pclass']].head(10)
# print(tab6)

# - Complete categorical feature: Embarked
# - Filling missing values with most common occurance
freq_port = train_df.Embarked.dropna().mode()[0]
# print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].replace(0,freq_port)
tab7 = train_df[['Embarked', 'Survived']].groupby(['Embarked'], \
                as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(tab7)
# - Convert categorical feature to numerical: Embarked
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# print(train_df.head())

# - Quick completing and converting numeric feature: Fare
# - We can now complete the Fare feature for single missing value in test dataset
# - using mode to get the value that occurs most frequently for this feature.
# - We may also want round off the fare to two decimals as it represents currency.
# =============================================================
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
# print(test_df.head())
# - Then we can create FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
tab8 = train_df[['FareBand', 'Survived']].groupby(['FareBand'], \
       as_index=False).mean().sort_values(by='FareBand', ascending=True)
# - Convert the Fare feature to ordinal values based on the FareBand.
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare' ] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare' ] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
# - Drop FareBand
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
# print(train_df.head(10))
# print(test_df.head(10))
# =============================================================


# --------------------- Part 2 --------------------- #
# # # # #      Model, predict and solve      # # # # #
# -------------------------------------------------- #
# Now we are ready to train a model and predict the required solution.
# Our problem is [Supervised Learning] + [Classification and Regression].
# we can narrow down our choice of models to a few. These include:
# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forrest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine
# Prepare data
# =============================================================
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
# print(X_train.shape, Y_train.shape, X_test.shape)
# =============================================================

# - - -    Logistic Regression    - - - #
# =============================================================
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
Y_pred = logReg.predict(X_test)
acc_log = round(logReg.score(X_train, Y_train) * 100, 2)
# print(acc_log)
# =============================================================
# # We can use Logistic Regression to validate our assumptions and decisions for
# feature creating and completing goals. This can be done by calculating the
# coefficient of the features in the decision function.
# Positive coefficients increase the log-odds of the response (and thus increase
# the probability), and negative coefficients decrease the log-odds of the
# response (and thus decrease the probability).
# - Sex is highest positivie coefficient, implying as the Sex value increases
#   (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - Inversely as Pclass increases, probability of Survived=1 decreases the most.
# - This way Age * Class is a good artificial feature to model as it has second
#   highest negative correlation with Survived.
# - So is Title as second highest positive correlation.
# =============================================================
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df['Correlation'] = pd.Series(logReg.coef_[0])
# print(coeff_df.sort_values(by='Correlation', ascending=False))
# =============================================================

# - - -    SVM    - - - #
# =============================================================
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print(acc_svc)
# =============================================================

# - - -   K-NN   - - - #
# =============================================================
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print(acc_knn)
# =============================================================

# - - -    Gaussian Naive Bayes   - - - #
# =============================================================
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print(acc_gaussian)
# =============================================================

# - - -   Perceptron   - - - #
# =============================================================
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print(acc_perceptron)
# =============================================================

# - - -   Linear SVC   - - - #
# =============================================================
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(acc_linear_svc)
# =============================================================

# - - -   Stochastic Gradient Descent   - - - #
# =============================================================
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print(acc_sgd)
# =============================================================

# - - -   Decision Tree   - - - #
# =============================================================
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)
# =============================================================

# - - -   Random Forest   - - - #
# =============================================================
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# print(acc_random_forest)
# =============================================================

# --------------------- Part 2 --------------------- #
# # # # #          Model Evaluation         # # # # #
# -------------------------------------------------- #
models = pd.DataFrame({
    'Model' : ['Support Vector Machine', 'KNN', 'Logistic Regression',
               'Random Forest', 'Naive Bays', 'Perceptron',
               'Stochastic Gradient Descent', 'Linear SVC',
               'Decision Tree'],
    'Score' : [acc_svc, acc_knn, acc_log,
               acc_random_forest, acc_gaussian, acc_perceptron,
               acc_sgd, acc_linear_svc,
               acc_decision_tree]})
# print(models.sort_values(by='Score', ascending=False))

# - - -   Submission - - - #
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
})
submission.drop(columns=0, inplace=False)
print(submission)
# submission.to_csv('t1.csv')
