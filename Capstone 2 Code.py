import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler


''' Data Cleansing '''
########################################################################################################################
########################################################################################################################

''' Import the railroad dataset and get rid of unnecessary columns '''
########################################################################################################################
raw_data = pd.read_excel('Railroad Dataset.xlsx')

# Create a copy of the data.
data = raw_data

# Rename the columns.
data.columns = ['Year of Incident', 'Month of Incident', 'Railroad Reporting', 'Accident/Incident Number',
                'Type of Person', 'Job Code', 'Nature of Injury', 'Location of Injury on Body',
                'Indicator of Death Within a Year', 'Old Casual Occurrence Code', 'Old Equipment Movement Indicator',
                'Age of Person Reported', 'Days Away From Work (Employee)',
                '# of Days of Restricted Activity (Employee)', 'Dummy', 'FIPS State Code', 'Railroad Class', 'Dummy1',
                'FRA Designated Region', 'Dummy2', 'Narrative Length', 'Fatality?', 'Form F6180-54 Filled?',
                'Form F6180-57 Filled?', 'Dummy3', 'Day of Incident', 'Year of Incident - 4 Digits', 'Hour of Incident',
                'Minute of Incident', 'AM or PM Indicator', 'County', 'County Code', 'FIPS and County Code',
                'Number of Positive Alcohol Tests', 'Number of Positive Drug Tests', 'Physical Act Circumstance Code',
                'General Location of Person at Time of Injury', 'On-track Equipment Involved',
                'Specific Location of Person At Time of Injury', 'Event Code', 'Additional Information About Injury',
                'Cause Code', 'Hazmat Exposure?', 'Employee Termination or Transfer?', 'Narrative1', 'Narrative2',
                'Narrative3', 'Covered Data (A, R, or P)', 'Latitude', 'Longitude']

# Drop the empty columns.
data = raw_data.drop(['Old Casual Occurrence Code', 'Old Equipment Movement Indicator',
                      'Dummy', 'Dummy1', 'Dummy2', 'Dummy3'], axis=1)

# Drop the useless columns.
data = data.drop(['Narrative1', 'Narrative2', 'Narrative3', 'Narrative Length', 'Accident/Incident Number'], axis=1)

# Remove columns that provide duplicate information.
data = data.drop(['Year of Incident', 'FIPS and County Code', 'County Code'], axis=1)

# Remove columns that provided more trouble than they were worth.
data = data.drop(['Latitude', 'Longitude'], axis=1)

# Change all the columns to strings so that we can remove the empty spaces.
data = data.apply(lambda x: x.astype(str).str.strip())

# Set all the empty values as nans.
data = data.replace('', np.nan)

# Create our target variable.
y = data[['Fatality?']]

# Encode the target variable.
le = LabelEncoder()
le.fit(y)
y = pd.DataFrame(le.transform(y))

''' Outlier Detection '''
########################################################################################################################

# Keep only the columns we think would help us answer our question.
data = data.drop(['Nature of Injury', 'Location of Injury on Body', 'Indicator of Death Within a Year',
                  'Days Away From Work (Employee)', '# of Days of Restricted Activity (Employee)', 'Fatality?',
                  'Additional Information About Injury', 'Employee Termination or Transfer?',
                  'Covered Data (A, R, or P)'], axis=1)

# Create the columns that we will treat as quantitative.
int_cols = ['Age of Person Reported', 'Day of Incident', 'Year of Incident - 4 Digits', 'Hour of Incident',
            'Minute of Incident', 'Month of Incident']

data[int_cols] = data[int_cols].apply(pd.to_numeric, errors='coerce')

# Impute the age of person reported with the average.
data['Age of Person Reported'] = data['Age of Person Reported'].fillna(round(data['Age of Person Reported'].mean()))

''' Feature Selction '''
# Drop the variables that could be explained by other variables.
data = data.drop(['County', 'Railroad Reporting', 'Job Code'], axis=1)

# Dummify the dataset.
data = pd.get_dummies(data)

''' Isolation Forests '''
# Initialize the model.
isof = IsolationForest()

# Fir the data and score it.
isof.fit(data)
outlier_scores = isof.predict(data)
outlier_scores = pd.DataFrame(outlier_scores)
outlier_scores.to_excel('Outlier Scores - IF.xlsx')

# Remove all the outliers
pd.DataFrame(data)
data['Outlier Score'] = outlier_scores

y = y.loc[data.loc[:, 'Outlier Score'] == 1]
data = data.loc[data.loc[:, 'Outlier Score'] == 1].drop('Outlier Score', axis=1)

''' Local Outlier Factor '''
# lof = LocalOutlierFactor()
#
# outlier_scores = pd.DataFrame(lof.fit_predict(data))
# outlier_scores.columns = ['Outlier Score']
# outlier_scores.to_excel('Outlier Scores - LOF.xlsx')


''' Dimensionality Reduction '''
########################################################################################################################

''' PCA '''
# Initialize the PCA model.
pca = PCA()

# Perform the PCA on the non-outlier data.
pca.fit_transform(data)

# Plot the results.
plt.plot(pca.explained_variance_)
plt.xlabel('n_components')
plt.ylabel('explained variance')
plt.title('Explained Variance by Variable - No Outliers')
plt.show()

# Print out the explained variance.
print(pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'], index=data.columns))

slim_data = data[['Month of Incident', 'Age of Person Reported', 'Day of Incident', 'Year of Incident - 4 Digits',
                  'Hour of Incident', 'Minute of Incident']]


''' Dealing with Unbalanced Classes '''
########################################################################################################################

# TODO: ''' Random Over-Sampling '''
# Increases the liklihood of overfitting
# http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler

# Using k-fold validation, break the data into test & train.  Then perform oversampling on the train set.
#


''' Machine Learning '''
########################################################################################################################
########################################################################################################################

# TODO Change the performance metrics to confusion matrix and ROC curves
# See https://elitedatascience.com/imbalanced-classes
# Initiate the scores dataframe.
scores = pd.DataFrame(columns=['Score'])

''' Logistic Regression '''
########################################################################################################################
# Initialize the model.
log_reg = LogisticRegression()

scores.loc['Logistic Regression', 'Score'] = np.mean(cross_val_score(log_reg, slim_data, y, cv=10))


# Trees are good for dealing with unbalanced data.
''' Decision Tree '''
########################################################################################################################
tree_classifer = DecisionTreeClassifier()

scores.loc['Decision Tree', 'Score'] = np.mean(cross_val_score(tree_classifer, slim_data, y, cv=10))

''' Gradient Boosted Classifier Tree '''
########################################################################################################################
gbc = GradientBoostingClassifier()

scores.loc['Gradient Boosting Classifier Tree', 'Score'] = np.mean(cross_val_score(gbc, slim_data, y, cv=10))

''' Random Forest '''
########################################################################################################################
rfc = RandomForestClassifier()

scores.loc['Random Forest Classifier', 'Score'] = np.mean(cross_val_score(rfc, slim_data, y, cv=10))

''' XG Boost '''
########################################################################################################################

# ''' Interpretation '''
# X_train, X_test, y_train, y_test = train_test_split(slim_data, y)
# log_reg.fit(X_train, y_train)
# print('The Coefficients for the logistic regression is: ')
# print(log_reg.coef_)
# print(slim_data.columns)


