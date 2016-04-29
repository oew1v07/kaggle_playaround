import numpy as np

def inv_dict(map_dict):
	"""Creates an inverse dictionary map"""
	inv_map = {v:k for k, v in map_dict.items()}
	return inv_map


gender_ord = {"FEMALE":0, "MALE":1, "OTHER":2, "-unknown-":3, -1:-1}

signup_method_ord = {'facebook':0, 'basic':1, 'google':2, -1:-1}

language_ord = {'en':0, 'fr':1, 'de':2, 'es':3, 'it':4, 'pt':5, 'zh':6, 'ko':7,
				'ja':8, 'ru':9, 'pl':10, 'el':11, 'sv':12, 'nl':13, 'hu':14,
				'da':15, 'id':16, 'fi':17, 'no':18, 'tr':19, 'th':20, 'cs':21,
				'hr':22, 'ca':23, 'is':24, -1:-1}

affiliate_channel_ord = {'direct':0, 'seo':1, 'other':2, 'sem-non-brand':3,
						 'content':4, 'sem-brand':5, 'remarketing':6, 'api':7, -1:-1}

affiliate_provider_ord = {'direct':0, 'google':1, 'other':2, 'craigslist':3,
						  'facebook':4, 'vast':5, 'bing':6, 'meetup':7,
						  'facebook-open-graph':8, 'email-marketing':9, 'yahoo':10,
						  'padmapper':11, 'gsp':12, 'wayn':13, 'naver':14,
						  'baidu':15, 'yandex':16, 'daum':17, -1:-1}

first_affiliate_tracked_ord = {'untracked':0, 'omg':1, 'linked':2, 
							   'tracked-other':3, 'product':4, 'marketing':5,
							   'local ops':6, -1:-1}

signup_app_ord = {'Web':0, 'Moweb':1, 'iOS':2, 'Android':3, -1:-1}

first_device_type_ord = {'Mac Desktop':0, 'Windows Desktop':1, 'iPhone':2, 
						 'Other/Unknown':3, 'Desktop (Other)':4, 'Android Tablet':5,
						 'iPad':6, 'Android Phone':7, 'SmartPhone (Other)':8, -1:-1}

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
					 'Stainless':48, 'Googlebot':49, 'Outlook 2007':50, 'IceDragon':51, -1:-1}

country_ord = {"NDF":0, "US":1, "other":2, "FR":3, "CA":4, "GB":5, "ES":6,
			   "IT":7, "PT":8, "NL":9, "DE":10, "AU":11}

country = inv_dict(country_ord)

# which cols to read in
default_cols = ["id", "date_account_created", "timestamp_first_active",
					  "date_first_booking", "gender", "age", "signup_method",
					  "signup_flow", "language", "affiliate_channel", 
					  "affiliate_provider", "first_affiliate_tracked", "signup_app",
					  "first_device_type", "first_browser"]

# remove the id field for the random forest
columns = np.array(["age", "signup_flow", "gender", "signup_method", "language",
					"affiliate_channel", "affiliate_provider", "first_affiliate_tracked",
					"signup_app", "first_device_type", "first_browser"])

