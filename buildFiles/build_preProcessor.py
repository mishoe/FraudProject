import pickle
import pandas as pd
from PreProcessor.feature_builder import DataSetBuilder
import xgboost as xgb
import numpy as np

#import
data=pd.read_csv('Data/train_sample.csv')

data['app']=data['app'].apply(str)
data['device']=data['device'].apply(str)
data['os']=data['os'].apply(str)
data['channel']=data['channel'].apply(str)

# set up the parameters
col_dict = {'cat_cols':['app','device','os','channel']}


# save the pickled data
file = 'data/raw_neighborhood_data.p'
pickle.dump(data, open(file, 'wb'))


# load the pickled data
data = pd.read_pickle('data/raw_neighborhood_data.p')




# Learn the preProcessing
trans = DataSetBuilder(col_dict=col_dict)
trans.fit(data)

# save the training data
features = xgb.DMatrix(trans.transform(data), feature_names=trans.feature_names)
xgb.DMatrix.save_binary(features, 'data/xgb.features.data')

# save the transform
file = 'data/transformer.p'
pickle.dump(trans, open(file, 'wb'))
#
#
# # load the  transform
# with open('data/transformer.p', 'rb') as f:
#     trans = pickle.load(f)
# # test on features
# features_test = trans.transform(data.head())
