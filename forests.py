import pandas as pd
import datetime as dt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

def inv_dict(map_dict):
	"""Creates an inverse dictionary map"""
	inv_map = {v:k for k, v in map_dict.items()}
	return inv_map

train_x = pd.read_csv("train_users_2.csv", parse_dates=[1,3],
					  usecols=["id", "date_account_created", "timestamp_first_active",
					  "date_first_booking", "gender", "age", "signup_method",
					  "signup_flow", "language", "affiliate_channel", 
					  "affiliate_provider", "first_affiliate_tracked", "signup_app",
					  "first_device_type", "first_browser"])
train_y = pd.read_csv("train_users_2.csv", usecols = ["country_destination"])
test = pd.read_csv("test_users.csv", parse_dates=[1,3])

# Now we need to create ordinal values for the categorical variables.
# Easiest way is to introduce a dict for each column

gender_ord = {"FEMALE":0, "MALE":1, "-unknown-":2}

signup_method_ord = {'facebook':0, 'basic':1, 'google':2}

language_ord = {'en':0, 'fr':1, 'de':2, 'es':3, 'it':4, 'pt':5, 'zh':6, 'ko':7,
				'ja':8, 'ru':9, 'pl':10, 'el':11, 'sv':12, 'nl':13, 'hu':14,
				'da':15, 'id':16, 'fi':17, 'no':18, 'tr':19, 'th':20, 'cs':21,
				'hr':22, 'ca':23, 'is':24}

affiliate_channel_ord = {'direct':0, 'seo':1, 'other':2, 'sem-non-brand':3,
						 'content':4, 'sem-brand':5, 'remarketing':6, 'api':7}

affiliate_provider_ord = {'direct':0, 'google':1, 'other':2, 'craigslist':3,
						  'facebook':4, 'vast':5, 'bing':6, 'meetup':7,
						  'facebook-open-graph':8, 'email-marketing':9, 'yahoo':10,
						  'padmapper':11, 'gsp':12, 'wayn':13, 'naver':14,
						  'baidu':15, 'yandex':16, 'daum':17}

first_affiliate_tracked_ord = {'untracked':0, 'omg':1, np.nan:2, 'linked':3, 
							   'tracked-other':4, 'product':5, 'marketing':6,
							   'local ops':7}

signup_app_ord = {'Web':0, 'Moweb':1, 'iOS':2, 'Android':3}

first_device_type_ord = {'Mac Desktop':0, 'Windows Desktop':1, 'iPhone':2, 
						 'Other/Unknown':3, 'Desktop (Other)':4, 'Android Tablet':5,
						 'iPad':6, 'Android Phone':7, 'SmartPhone (Other)':8}

first_browser_ord = {'Chrome':0, 'IE':1, 'Firefox':2, 'Safari':3, '-unknown-':4,
					 'Mobile Safari':5, 'Chrome Mobile':6, 'RockMelt':7,
					 'Chromium':8, 'Android Browser':9, 'AOL Explorer':10,
					 'Palm Pre web browser':11, 'Mobile Firefox':12, 'Opera':13,
					 'TenFourFox':14, 'IE Mobile':15, 'Apple Mail':16, 'Silk':17,
					 'Camino':18, 'Arora':19, 'BlackBerry Browser':20, 'SeaMonkey':21,
					 'Iron':22, 'Sogou Explorer':23, 'IceWeasel':24, 'Opera Mini':25,
					 'SiteKiosk':26, 'Maxthon':27, 'Kindle Browser':28,
					 'CoolNovo':29, 'Conkeror':30, 'wOSBrowser':31, 'Google Earth':32,
					 'Crazy Browser':33, 'Mozilla':34, 'OmniWeb':35, 
					 'PS Vita browser':36, 'NetNewsWire':37, 'CometBird':38,
					 'Comodo Dragon':39, 'Flock':40, 'Pale Moon':41,
					 'Avant Browser':42, 'Opera Mobile':43, 'Yandex.Browser':44,
					 'TheWorld Browser':45, 'SlimBrowser':46, 'Epic':47,
					 'Stainless':48, 'Googlebot':49, 'Outlook 2007':50, 'IceDragon':51}

country_ord = {"NDF":0, "US":1, "other":2, "FR":3, "CA":4, "GB":5, "ES":6,
			   "IT":7, "PT":8, "NL":9, "DE":10, "AU":11}

country = inv_dict(country_ord)

# to actually map these we do dataframe.map(dict)
train_x["gender"] = train_x.gender.map(gender_ord)
train_x["signup_method"] = train_x.signup_method.map(signup_method_ord)
train_x["language"] = train_x.language.map(language_ord)
train_x["affiliate_channel"] = train_x.affiliate_channel.map(affiliate_channel_ord)
train_x["affiliate_provider"] = train_x.affiliate_provider.map(affiliate_provider_ord)
train_x["first_affiliate_tracked"] = train_x.first_affiliate_tracked.map(first_affiliate_tracked_ord)
train_x["signup_app"] = train_x.signup_app.map(signup_app_ord)
train_x["first_device_type"] = train_x.first_device_type.map(first_device_type_ord)
train_x["first_browser"] = train_x.first_browser.map(first_browser_ord)
train_y["country_destination"] = train_y.country_destination.map(country_ord)

test["gender"] = test.gender.map(gender_ord)
test["signup_method"] = test.signup_method.map(signup_method_ord)
test["language"] = test.language.map(language_ord)
test["affiliate_channel"] = test.affiliate_channel.map(affiliate_channel_ord)
test["affiliate_provider"] = test.affiliate_provider.map(affiliate_provider_ord)
test["first_affiliate_tracked"] = test.first_affiliate_tracked.map(first_affiliate_tracked_ord)
test["signup_app"] = test.signup_app.map(signup_app_ord)
test["first_device_type"] = test.first_device_type.map(first_device_type_ord)
test["first_browser"] = test.first_browser.map(first_browser_ord)

# remove the id field
columns = np.array(["age", "timestamp_first_active", "signup_flow", "gender", 
					"signup_method", "language", "affiliate_channel",
					"affiliate_provider", "first_affiliate_tracked", "signup_app",
					"first_device_type", "first_browser"])

# xtrain and xtest is the dataframe with the id taken out
xtrain = train_x[columns]
xtest = test[columns]

imp = Imputer(strategy="median")
xtrain_na = imp.fit_transform(xtrain)
xtest_na = imp.fit_transform(xtest)


def rforests(trainx, trainy, test, n_estimators=100, standardize=True):
	trainy = np.ravel(trainy)
	if standardize:
		# does it even make sense to standardize these values?
		means = np.mean(trainx, axis=0)
		trainx = (trainx.T / means[:,None]).T

	forest = RandomForestClassifier(n_estimators)
	forest.fit(trainx, trainy)
	output = forest.predict(test)
	df_out = pd.DataFrame(output, columns=["prediction"])
	df_out = df_out.prediction.map(country)
	return forest, df_out

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

	return score, forest

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


def make_sub(df_out, test, name = "my"):
	sub = pd.DataFrame([test.id, df_out], index=["id", "country"])
	submission = sub.T
	subname = name + "submission.csv"
	submission.to_csv(subname, index=False)
	return submission


# forest, output = rforests(xtrain_na, train_y, xtest_na)

