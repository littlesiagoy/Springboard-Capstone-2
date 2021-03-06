import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
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

''' Feature Selction '''
#######################################################################################################################
# Drop the variables that could be explained by other variables.
data = data.drop(['County', 'Railroad Reporting', 'Job Code'], axis=1)

# Keep only the columns we think would help us answer our question.
data = data.drop(['Nature of Injury', 'Location of Injury on Body', 'Indicator of Death Within a Year',
                  'Days Away From Work (Employee)', '# of Days of Restricted Activity (Employee)', 'Fatality?',
                  'Additional Information About Injury', 'Employee Termination or Transfer?',
                  'Covered Data (A, R, or P)'], axis=1)

# Create the columns that we will treat as quantitative.
int_cols = ['Age of Person Reported', 'Day of Incident', 'Year of Incident - 4 Digits', 'Hour of Incident',
            'Minute of Incident', 'Month of Incident']

# Convert those columns as ints.
data[int_cols] = data[int_cols].apply(pd.to_numeric, errors='coerce')

# Dummify the dataset.
data = pd.get_dummies(data)
#######################################################################################################################
#######################################################################################################################


''' Test/Train Split, Outlier Detection, and Model Building '''
#######################################################################################################################
#######################################################################################################################

# Create the time series split instance.
tscv = TimeSeriesSplit(n_splits=3)

# Initalize the scores dataframe, and index.
scores = pd.DataFrame()
scores2 = pd.DataFrame()
scores3 = pd.DataFrame()
scores4 = pd.DataFrame()

index = 0

''' Perform the Splitting, Outlier Detection, and Model Training'''
# Actually use the time series split to create the test and training sets.
for train_index, test_index in tscv.split(data):

    ''' Train & Test Split '''
    # Create the train & test split.
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ''' Data Imputation '''
    # Impute the age of person reported with the average.
    X_train['Age of Person Reported'] = X_train['Age of Person Reported'].fillna(round(X_train['Age of Person Reported'].mean()))
    X_test['Age of Person Reported'] = X_test['Age of Person Reported'].fillna(round(X_test['Age of Person Reported'].mean()))

    ''' Isolation Forests '''
    # Initialize the model.
    isof = IsolationForest()

    # Fit the data and score it.
    isof.fit(X_train)
    outlier_scores = isof.predict(X_train)
    outlier_scores = pd.DataFrame(outlier_scores)
    outlier_scores.to_excel('Outlier Scores - IF.xlsx')

    ''' Local Outlier Factor '''
    # lof = LocalOutlierFactor()
    #
    # outlier_scores_lof = pd.DataFrame(lof.fit_predict(X_train))
    # outlier_scores_lof.columns = ['Outlier Score']
    # outlier_scores_lof.to_excel('Outlier Scores - LOF.xlsx')

    # Remove all the outliers
    pd.DataFrame(X_train)
    X_train['Outlier Score'] = outlier_scores

    y_train = y_train.loc[X_train.loc[:, 'Outlier Score'] == 1]
    X_train = X_train.loc[X_train.loc[:, 'Outlier Score'] == 1].drop('Outlier Score', axis=1)

    # Delete the isolation forest for memory.
    del isof

    ''' Perform No PCA & No Oversampling, and then train the models. '''
    ####################################################################################################################
    ''' Model Training '''
    # Logistic Regression
    log_reg = LogisticRegression()

    # Decision Trees
    tree_classifer = DecisionTreeClassifier()

    # Random Forest
    rfc = RandomForestClassifier()

    # Fit the data using the various models.
    log_reg.fit(X_train, y_train)
    tree_classifer.fit(X_train, y_train)
    rfc.fit(X_train, y_train)

    # Predict with the various models.
    y_log_reg = log_reg.predict(X_test)
    y_tree_classifier = tree_classifer.predict(X_test)
    y_rfc = rfc.predict(X_test)

    # Compute the AROC from the predictions.
    scores.loc[index, 'Logistic Regresion'] = roc_auc_score(y_test, y_log_reg)
    scores.loc[index, 'Decision Tree'] = roc_auc_score(y_test, y_tree_classifier)
    scores.loc[index, 'Random Forest'] = roc_auc_score(y_test, y_rfc)

    ''' Perform No PCA & Oversampling, then Model training '''
    ####################################################################################################################

    ''' Perform Oversampling '''
    # Create the oversampling instance.
    ros = RandomOverSampler(random_state=2018)

    # Oversample the training set.
    X_train, y_train = ros.fit_sample(X_train, y_train)

    ''' Model Training '''
    # Logistic Regression
    log_reg = LogisticRegression()

    # Decision Trees
    tree_classifer = DecisionTreeClassifier()

    # Random Forest
    rfc = RandomForestClassifier()

    # Fit the data using the various models.
    log_reg.fit(X_train, y_train)
    tree_classifer.fit(X_train, y_train)
    rfc.fit(X_train, y_train)

    # Predict with the various models.
    y_log_reg = log_reg.predict(X_test)
    y_tree_classifier = tree_classifer.predict(X_test)
    y_rfc = rfc.predict(X_test)

    # Compute the AROC from the predictions.
    scores2.loc[index, 'Logistic Regresion'] = roc_auc_score(y_test, y_log_reg)
    scores2.loc[index, 'Decision Tree'] = roc_auc_score(y_test, y_tree_classifier)
    scores2.loc[index, 'Random Forest'] = roc_auc_score(y_test, y_rfc)


    ''' Reset the data to what it was like before Oversampling before performing PCA & Oversampling. '''
    ####################################################################################################################
    ''' Train & Test Split '''
    # Create the train & test split again.
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ''' Data Imputation '''
    # Impute the age of person reported with the average.
    X_train['Age of Person Reported'] = X_train['Age of Person Reported'].fillna(
        round(X_train['Age of Person Reported'].mean()))
    X_test['Age of Person Reported'] = X_test['Age of Person Reported'].fillna(
        round(X_test['Age of Person Reported'].mean()))

    ''' Remove Outliers '''
    # Remove all the outliers
    pd.DataFrame(X_train)
    X_train['Outlier Score'] = outlier_scores

    y_train = y_train.loc[X_train.loc[:, 'Outlier Score'] == 1]
    X_train = X_train.loc[X_train.loc[:, 'Outlier Score'] == 1].drop('Outlier Score', axis=1)

    # Delete the outlier_scores for memeory.
    del outlier_scores

    ''' Perform PCA & No Oversampling '''
    ####################################################################################################################
    ''' PCA '''
    # Initialize the PCA model.
    pca = PCA()

    # Perform the PCA on the non-outlier data.
    pca.fit_transform(X_train)

    # Plot the results.
    # plt.plot(pca.explained_variance_)
    # plt.xlabel('n_components')
    # plt.ylabel('explained variance')
    # plt.title('Explained Variance by Variable - No Outliers')
    # plt.show()

    # Print out the explained variance.
    print(pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'], index=data.columns))

    # Slim down the data based on the PCA results.
    X_train = X_train[['Month of Incident', 'Age of Person Reported', 'Day of Incident', 'Year of Incident - 4 Digits',
                       'Hour of Incident', 'Minute of Incident']]

    X_test = X_test[['Month of Incident', 'Age of Person Reported', 'Day of Incident', 'Year of Incident - 4 Digits',
                     'Hour of Incident', 'Minute of Incident']]

    ''' Model Training '''
    # Logistic Regression
    log_reg = LogisticRegression()

    # Decision Trees
    tree_classifer = DecisionTreeClassifier()

    # Random Forest
    rfc = RandomForestClassifier()

    # Fit the data using the various models.
    log_reg.fit(X_train, y_train)
    tree_classifer.fit(X_train, y_train)
    rfc.fit(X_train, y_train)

    # Predict with the various models.
    y_log_reg = log_reg.predict(X_test)
    y_tree_classifier = tree_classifer.predict(X_test)
    y_rfc = rfc.predict(X_test)

    # Compute the AROC from the predictions.
    scores3.loc[index, 'Logistic Regresion'] = roc_auc_score(y_test, y_log_reg)
    scores3.loc[index, 'Decision Tree'] = roc_auc_score(y_test, y_tree_classifier)
    scores3.loc[index, 'Random Forest'] = roc_auc_score(y_test, y_rfc)

    ''' Perform PCA & Oversampling '''
    ####################################################################################################################
    # Create the oversampling instance.
    ros = RandomOverSampler(random_state=2018)

    # Oversample the training set.
    X_train, y_train = ros.fit_sample(X_train, y_train)

    ''' Model Training '''
    # Logistic Regression
    log_reg = LogisticRegression()

    # Decision Trees
    tree_classifer = DecisionTreeClassifier()

    # Random Forest
    rfc = RandomForestClassifier()

    # Fit the data using the various models.
    log_reg.fit(X_train, y_train)
    tree_classifer.fit(X_train, y_train)
    rfc.fit(X_train, y_train)

    # Predict with the various models.
    y_log_reg = log_reg.predict(X_test)
    y_tree_classifier = tree_classifer.predict(X_test)
    y_rfc = rfc.predict(X_test)

    # Compute the AROC from the predictions.
    scores4.loc[index, 'Logistic Regresion'] = roc_auc_score(y_test, y_log_reg)
    scores4.loc[index, 'Decision Tree'] = roc_auc_score(y_test, y_tree_classifier)
    scores4.loc[index, 'Random Forest'] = roc_auc_score(y_test, y_rfc)

    # Increase the iterator
    index += 1


# Export the results
writer = pd.ExcelWriter('Training Results.xlsx', engine='xlsxwriter', datetime_format='mm/dd/yyyy')
workbook = writer.book

scores.to_excel(writer, sheet_name='No PCA & No Oversampling')
scores2.to_excel(writer, sheet_name='No PCA & Oversampling')
scores3.to_excel(writer, sheet_name='PCA & No Oversampling')
scores4.to_excel(writer, sheet_name='PCA & Oversampling')
workbook.close()