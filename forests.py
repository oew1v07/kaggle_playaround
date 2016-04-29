import pandas as pd
import datetime as dt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from rank_metrics import ndcg_at_k
import dicts


def transform_data(train_path="train_users_2.csv", test_path="test_users.csv",
				   xcols=dicts.default_cols, ycols=["country_destination"], parse_dates=[1,3],
				   final_cols = dicts.columns):
	"""Reads in data, splits into x and target and then transform into numeric
	data using the supplied dicts from dicts.py.
	"""

	trainx = pd.read_csv(train_path, parse_dates=parse_dates, usecols=xcols)
	trainy = pd.read_csv(train_path, usecols = ycols)
	test = pd.read_csv(test_path, parse_dates=parse_dates, usecols=xcols)
	trainx = trainx.fillna(-1)
	test = test.fillna(-1)

	# Now we need to create ordinal values for the categorical variables.
	# Easiest way is to introduce a dict for each column

	# to actually map these we do dataframe.map(dict)
	if xcols == dicts.default_cols:
		trainx["gender"] = trainx.gender.map(dicts.gender_ord)
		trainx["signup_method"] = trainx.signup_method.map(dicts.signup_method_ord)
		trainx["language"] = trainx.language.map(dicts.language_ord)
		trainx["affiliate_channel"] = trainx.affiliate_channel.map(dicts.affiliate_channel_ord)
		trainx["affiliate_provider"] = trainx.affiliate_provider.map(dicts.affiliate_provider_ord)
		trainx["first_affiliate_tracked"] = trainx.first_affiliate_tracked.map(dicts.first_affiliate_tracked_ord)
		trainx["signup_app"] = trainx.signup_app.map(dicts.signup_app_ord)
		trainx["first_device_type"] = trainx.first_device_type.map(dicts.first_device_type_ord)
		trainx["first_browser"] = trainx.first_browser.map(dicts.first_browser_ord)

		trainy["country_destination"] = trainy.country_destination.map(dicts.country_ord)

		test["gender"] = test.gender.map(dicts.gender_ord)
		test["signup_method"] = test.signup_method.map(dicts.signup_method_ord)
		test["language"] = test.language.map(dicts.language_ord)
		test["affiliate_channel"] = test.affiliate_channel.map(dicts.affiliate_channel_ord)
		test["affiliate_provider"] = test.affiliate_provider.map(dicts.affiliate_provider_ord)
		test["first_affiliate_tracked"] = test.first_affiliate_tracked.map(dicts.first_affiliate_tracked_ord)
		test["signup_app"] = test.signup_app.map(dicts.signup_app_ord)
		test["first_device_type"] = test.first_device_type.map(dicts.first_device_type_ord)
		test["first_browser"] = test.first_browser.map(dicts.first_browser_ord)

		# xtrain and xtest is the dataframe with the id taken out
		xtrain = trainx[dicts.columns]
		xtest = test[dicts.columns]

		return xtrain, trainy, xtest


def rforests(trainx, trainy, test, n_estimators=100, k=5):
	trainy = np.ravel(trainy)

	forest = RandomForestClassifier(n_estimators)
	forest.fit(trainx, trainy)


	prob_train = forest.predict_proba(trainx)
	prob_test = forest.predict_proba(test)

	# Since the index is the number of the country that's been chosen
	# we can use these with argsort to get the maximum 5., we will have to do this
	# for the entire matrix though.
	sort_train = np.argsort(prob_train)[:,-k:]
	sort_test = np.argsort(prob_test)[:,-k:]

	# Now we need to transform these back to countries, but to map I need to
	# have a dataframe.
	col_names = []

	for i in range(k):
		name = "country_destination_" + str(i+1)
		col_names.append(name)

	pred_train = pd.DataFrame(sort_train, columns=col_names)
	pred_test = pd.DataFrame(sort_test, columns=col_names)

	for name in col_names:
		pred_train[name] = pred_train[name].map(country)
		pred_test[name] = pred_test[name].map(country)

	return forest, pred_train, pred_test

def cross_val(trainx, trainy, n_estimators=100, test_split=0.4, num_cross=10):
	score = np.zeros((1,num_cross))

	for i in range(num_cross):
		X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, 
															test_size=test_split)
		forest = RandomForestClassifier(n_estimators)

		# Making the column vectors into 1d arrays
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
		forest.fit(X_train, y_train)
		score[0, i] = forest.score(X_test, y_test)

	return score, prediction

def benchmark_countries(trainy, country="NDF"):
	"""Finding the accuracy of just using one country or NDF from the training data"""
	# Because we've encoded 
	if country == "NDF":
		pred = np.zeros(trainy.shape)

	else:
		pred = country_ord[country]*np.ones(trainy.shape)

	comp = pred == trainy

	acc = comp.sum()/comp.count()

	return acc

def calc_NDCG(prediction, target, k=5):
	"""This needs to work out the relevances and once the NDCG is calculated
	then all of the NDCG for each query is added together and divided by 
	the total amount of queries giving a mean NDCG score."""
	
	rel = prediction == target
	all_NDCG = np.zeros(target.shape)

	for i, relevance in enumerate(rel):
		all_NDCG[i,0] = ndcg_at_k(relevance, k, 1)

	NDCG = all_NDCG.sum()/len(all_NDCG)

	return NDCG


def make_sub(prediction, test, name = "my"):
	# Needs to count the number of columns prediction has.
	count = prediction.shape[1]

	sub = pd.DataFrame([test.id,np.zeros(test.id.shape)], index=["id", "country"])
	submission = sub.T
	
	for i, pred in enumerate(prediction):
		submission["country"][i] = list(prediction)

	subname = name + "submission.csv"
	submission.to_csv(subname, index=False)
	return submission


# forest, output = rforests(xtrain_na, train_y, xtest_na)

