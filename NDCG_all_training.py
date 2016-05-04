import pandas as pd
import numpy as np
import dicts
from forests import *

####################
# User hot encoded #
####################

trainx, trainy, test = transform_data(train_path="data_train_users_hot.csv",
									  test_path="data_test_users_hot.csv",xcols=dicts.users_hot_encode_cols,
									  parse_dates=None,final_cols=None)

pred_train = pd.read_csv("pred_train_users_hot.csv")
pred_test = pd.read_csv("pred_test_users_hot.csv")

out1, NDCG1 = confusion_matrix(pred_train, trainy)

#################
# User no dates #
#################

trainx, trainy, test = transform_data(train_path="data_train_users_2.csv",
									  test_path="data_test_users.csv",xcols=dicts.default_cols,
									  parse_dates=[1,3],final_cols=dicts.columns)

pred_train = pd.read_csv("pred_train_users_nodates.csv")
pred_test = pd.read_csv("pred_test_users_nodates.csv")

out2, NDCG2 = confusion_matrix(pred_train, trainy)

##############################
# User hot encoded + session #
##############################

trainx, trainy, test = transform_data(train_path="data_train_users_sess.csv",
									  test_path="data_test_users_sess.csv",xcols=dicts.users_sessions_cols,
									  parse_dates=None,final_cols=None)

pred_train = pd.read_csv("pred_train_users_sess.csv")
pred_test = pd.read_csv("pred_test_users_sess.csv")

out3, NDCG3 = confusion_matrix(pred_train, trainy)


