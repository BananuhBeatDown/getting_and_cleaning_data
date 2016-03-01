# import needed libraries
import os
import pandas as pd
import numpy as np

# change to proper directory
os.chdir('C:\Users\Matt\Desktop\Python Files\Getting and Cleaning Data Project')

# open the test and training files
subjectTest = pd.read_csv('subject_test.txt', names=['subject ID'])
subjectTrain = pd.read_csv('subject_train.txt', names=['subject ID'])
xTest = pd.read_fwf('X_test.txt', header=None)
xTrain = pd.read_fwf('X_train.txt', header=None)
yTest = pd.read_fwf('Y_test.txt', names=['activity ID'])
yTrain = pd.read_fwf('Y_train.txt', names=['activity ID'])

# concat the test and training files
subjectConcat = pd.concat([subjectTest, subjectTrain], ignore_index=True)
xConcat = pd.concat([xTest, xTrain], ignore_index=True)
yConcat = pd.concat([yTest, yTrain], ignore_index=True)

# open the activity and feature files
activityLabels = pd.read_fwf('activity_labels.txt', names=['activity ID', 'activity'])
features = pd.read_csv('features.txt', names=['features'])

# clean feautures by creating a list of the features the mean and std features 
cleanFeatures = features[features['features'].str.contains(r'mean()\b|std()\b')]
cleanFeatures = pd.Series(cleanFeatures['features'])
cleanFeatures = pd.DataFrame(cleanFeatures.str.split(' ').str.get(1))

# concat the x values with their matching features values(ints) as an index
xClean = xConcat.T.reindex(index=cleanFeatures.index).T

# merge the y values with their corresponding 'activity ID' from activityLabels
yClean = pd.merge(yConcat, activityLabels, on=['activity ID'])

# merge the cleaned x value with the cleaned y value
cleanData = pd.concat([yClean['activity'], xClean], axis=1)

# change the data frame long and narrow from short and wide 
cleanData = pd.melt(cleanData, id_vars=['activity'])

# merge the cleanData variable with the cleanFeatutes index and drop the variable column
cleanData = pd.merge(cleanData, cleanFeatures, left_on=['variable'], right_index=True, how='inner')
cleanData = cleanData.drop(['variable'], axis=1)

# creating separate columns for each variable
cleanData['phase'] = np.where(cleanData[0].str.contains(r'^t'), ['time'], ['four'])
cleanData['signal'] = np.where(cleanData[0].str.contains(r'Body'), ['body'], ['grav'])
cleanData['device'] = np.where(cleanData[0].str.contains(r'Acc'), ['acc'], ['gyro'])
cleanData['jerk'] = np.where(cleanData[0].str.contains(r'Jerk'), ['jerk'], [None])
cleanData['mag'] = np.where(cleanData[0].str.contains(r'Mag'), ['mag'], [None])
cleanData['func'] = np.where(cleanData[0].str.contains(r'mean()'), ['mean'], ['std'])
cleanData['domain'] = np.where(cleanData[0].str.contains(r'X$'), ['X'], np.where(cleanData[0].str.contains(r'Y$'), ['Y'], np.where(cleanData[0].str.contains(r'Z$'), ['Z'], [None])))
cleanData = cleanData.drop([0], axis=1)
cleanData = cleanData[['activity', 'phase', 'signal', 'device', 'jerk', 'mag', 'func', 'domain', 'value']]

print cleanData