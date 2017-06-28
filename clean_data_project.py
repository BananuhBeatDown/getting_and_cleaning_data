# -*- coding: utf-8 -*-
"""
Created September 1 10:13:47 2016

@author: Matt Green
"""

# import needed libraries
import pandas as pd
import numpy as np
import pickle
import os

# open the test and training files
subjectTest = pd.read_csv('subject_test.txt', names=['subject ID'])
subjectTrain = pd.read_csv('subject_train.txt', names=['subject ID'])
xTest = pd.read_fwf('X_test.txt', header=None)
xTrain = pd.read_fwf('X_train.txt', header=None)
yTest = pd.read_fwf('Y_test.txt', names=['activity ID'])
yTrain = pd.read_fwf('Y_train.txt', names=['activity ID'])

# %%

# concat the test and training files
subjectConcat = pd.concat([subjectTest, subjectTrain], ignore_index=True)
xConcat = pd.concat([xTest, xTrain], ignore_index=True)
yConcat = pd.concat([yTest, yTrain], ignore_index=True)

# %%

# open the activity and feature files
activityLabels = pd.read_fwf('activity_labels.txt', names=['activity ID', 'activity'])
features = pd.read_csv('features.txt', names=['features'])

# %%

# clean feautures by creating a list of the features the mean and std features 
cleanFeatures = features[features['features'].str.contains(r'mean()\b|std()\b')]
cleanFeatures = pd.Series(cleanFeatures['features'])
cleanFeatures = pd.DataFrame(cleanFeatures.str.split(' ').str.get(1))

# %%

# concat the x values with their matching features values(ints) as an index
xClean = xConcat.T.reindex(index=cleanFeatures.index).T

# %%

# merge the y values with their corresponding 'activity ID' from activityLabels
yClean = pd.merge(yConcat, activityLabels, on=['activity ID'])

# %%

# merge the cleaned x value with the cleaned y value
cleanData = pd.concat([yClean['activity'], xClean], axis=1)

# %%

# change the data frame long and narrow from short and wide 
cleanData = pd.melt(cleanData, id_vars=['activity'])

# %%

# merge the cleanData variable with the cleanFeatutes index and drop the variable column
cleanData = pd.merge(cleanData, cleanFeatures, left_on=['variable'], right_index=True, how='inner')
cleanData = cleanData.drop(['variable'], axis=1)

# %%

# create helper function for creating new features
def cleanData_feat(identifier, tag):
    return np.where(cleanData['features'].str.contains(identifier), tag[0], tag[1])

cleanData['phase'] = cleanData_feat(r'^t', ['time', 'four'])
cleanData['signal'] = cleanData_feat(r'Body', ['body', 'grav'])
cleanData['device'] = cleanData_feat(r'Acc', ['acc', 'gyro'])
cleanData['jerk'] = cleanData_feat(r'Jerk', ['jerk', None])
cleanData['mag'] = cleanData_feat(r'Mag', ['mag', None])
cleanData['func'] = cleanData_feat(r'mean()', ['mean', 'std'])
cleanData['domain'] = cleanData_feat(r'X$', ['X', cleanData_feat(r'Y$', ['Y', cleanData_feat(r'Z$', ['Z', None])])])

# delete the features column
cleanData = cleanData.drop(['features'], axis=1)

# reorganize the data
cleanData = cleanData[['activity', 'phase', 'signal', 'device', 'jerk', 'mag', 'func', 'domain', 'value']]

# %%

# Save the datasets as individually
# labelled features in a pickle file. 

pickle_file = 'clean_dataset.pickle'
f = open(pickle_file, 'wb')

pickle.dump(cleanData, f, pickle.HIGHEST_PROTOCOL)

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
