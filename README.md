# Getting and Cleaning Data with Python

This program creates dataset by collecting and cleaning data from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). 

## Install

- Python 3.6
    + I recommend installing [Anaconda](https://www.continuum.io/downloads) as it is already set-up with Pandas and other required libraries
    + If unfamiliar with the command line there are graphical installs for macOS, Windows, and Linux

## Dataset

Experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) wearing a smartphone (Samsung Galaxy S II) on the waist.

For each record in the dataset it is provide:
- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration. 
- Triaxial Angular velocity from the gyroscope. 
- A 561-feature vector with time and frequency domain variables. 
- Its activity label. 
- An identifier of the subject who carried out the experiment.

## Cleaning Procedure

- Concat the training and testing data
- Create a list of unique mean and standard deviation features
- Concat the input values using their matching feature values as an index
- Merge the output values with their corresponding activity
- Concat the input and output values
- Melt the data into a long and narrow format
- Merge the data with their features
- Create activity features based on `features` column descriptions
- Delete `features` column from the dataframe
- Reorganize the data into an easy to read format

## Run

`clean_data_project.py`

## Example Output

the terminal output:

![alt text](https://user-images.githubusercontent.com/10539813/27749869-b1ac0cd8-5dd5-11e7-97dd-a751e617982d.png)

the clean dataset:

<img src="https://user-images.githubusercontent.com/10539813/27749604-87fc6b04-5dd4-11e7-846a-35cb0abb2cb0.png" width="512">

## License

The getting_and_cleaning_data program is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). I encourage you to use it, and enhance your understand of the pandas library and sound data cleaning practice. :) 
