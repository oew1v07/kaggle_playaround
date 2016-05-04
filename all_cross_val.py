import pandas as pd
import numpy as np
import dicts
from forests import *

maximum = {"user_hot":142, "user_nodates":150, "user_hot_sess":142}

####################
# User hot encoded #
####################

trainx, trainy, test = transform_data(train_path="data_train_users_hot.csv",
									  test_path="data_test_users_hot.csv",xcols=dicts.users_hot_encode_cols,
									  parse_dates=None,final_cols=None)

df = pd.read_csv("data_train_users_hot.csv")
df_id = df.id
df = pd.read_csv("data_test_users_hot.csv")
df_test_id = df.id

forest, pred_train, pred_test = rforests(trainx, trainy, test, n_estimators=maximum["user_hot"], k=5)

pred_train = pd.DataFrame(pred_train, index=df_id)
pred_train.to_csv("pred_train_users_hot.csv")
pred_test = pd.DataFrame(pred_test, index=df_test_id)
pred_test.to_csv("pred_test_users_hot.csv")

prob_train = forest.predict_proba(trainx)
df_prob_train = pd.DataFrame(prob_train, index = df_id)
df_prob_train.to_csv("prob_train_user_hot.csv")
prob_test = forest.predict_proba(test)
df_prob_test = pd.DataFrame(prob_test, index = df_test_id)
df_prob_test.to_csv("prob_test_user_hot.csv")

#################
# User no dates #
#################

trainx, trainy, test = transform_data(train_path="data_train_users_2.csv",
									  test_path="data_test_users.csv",xcols=dicts.default_cols,
									  parse_dates=[1,3],final_cols=dicts.columns)

df = pd.read_csv("data_train_users_2.csv")
df_id = df.id
df = pd.read_csv("data_test_users.csv")
df_test_id = df.id

forest, pred_train, pred_test = rforests(trainx, trainy, test, n_estimators=maximum["user_nodates"], k=5)

pred_train = pd.DataFrame(pred_train, index=df_id)
pred_train.to_csv("pred_train_users_nodates.csv")
pred_test = pd.DataFrame(pred_test, index=df_test_id)
pred_test.to_csv("pred_test_users_nodates.csv")

prob_train = forest.predict_proba(trainx)
df_prob_train = pd.DataFrame(prob_train, index= df_id)
df_prob_train.to_csv("prob_train_user_nodates.csv")
prob_test = forest.predict_proba(test)
df_prob_test = pd.DataFrame(prob_test, index = df_test_id)
df_prob_test.to_csv("prob_test_user_nodates.csv")

##############################
# User hot encoded + session #
##############################

trainx, trainy, test = transform_data(train_path="data_train_users_sess.csv",
									  test_path="data_test_users_sess.csv",xcols=dicts.users_sessions_cols,
									  parse_dates=None,final_cols=None)

df = pd.read_csv("data_train_users_sess.csv")
df_id = df.id
df = pd.read_csv("data_test_users_sess.csv")
df_test_id = df.id

forest, pred_train, pred_test = rforests(trainx, trainy, test, n_estimators=maximum["user_hot_sess"], k=5)

pred_train = pd.DataFrame(pred_train, index=df_id)
pred_train.to_csv("pred_train_users_sess.csv")
pred_test = pd.DataFrame(pred_test, index=df_test_id)
pred_test.to_csv("pred_test_users_sess.csv")

prob_train = forest.predict_proba(trainx)
df_prob_train = pd.DataFrame(prob_train, index = df_id)
df_prob_train.to_csv("prob_train_users_sess.csv")
prob_test = forest.predict_proba(test)
df_prob_test = pd.DataFrame(prob_test, index = df_test_id)
df_prob_test.to_csv("prob_test_users_sess.csv")
