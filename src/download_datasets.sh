#!/bin/bash

# mkdir data

# Download and extract human activity data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip -d data/har
rm UCI\ HAR\ Dataset.zip


# For more, check 
# https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/download_datasets.sh