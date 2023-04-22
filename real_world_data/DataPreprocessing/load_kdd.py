from __future__ import division
# import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def load_kdd():
	FEATURES_CLASSIFICATION = ["age","class-of-worker", "detailed-industry-recode", "detailed-occupation-recode",
	"education", "wage-per-hour", "enroll-in-edu-inst-last-wk", "marital-stat", "major-industry-code",
	"major-occupation-code", "race", "hispanic-origin", "sex", "member-of-a-labor-union", "reason-for-unemployment",
	"full-or-part-time-employment-stat", "capital-gains", "capital-losses", "dividends-from-stocks", "tax-filer-stat",
	"region-of-previous-residence", "state-of-previous-residence", "detailed-household-and-family-stat",
	"detailed-household-summary-in-household", "migration-code-change-in-msa", "migration-code-change-in-reg",
	"migration-code-move-within-reg", "live-in-this-house-1-year-ago", "migration-prev-res-in-sunbelt",
	"num-persons-worked-for-employer", "family-members-under-18", "country-of-birth-father", "country-of-birth-mother",
	"country-of-birth-self", "citizenship", "own-business-or-self-employed", "fill-inc-questionnaire-for-veterans-admin", "veterans-benefits",
	"weeks-worked-in-year", "year"]

	# CONT_VARIABLES = ["capital-gains", "capital-losses", 'age'] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CONT_VARIABLES = ["age", "detailed-industry-recode","detailed-occupation-recode","wage-per-hour","capital-gains", "capital-losses", "num-persons-worked-for-employer", "dividends-from-stocks", "veterans-benefits","weeks-worked-in-year", "year",] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot


	CLASS_FEATURE = "class" # the decision variable
	SENSITIVE_ATTRS = ["sex"]


	# COMPAS_INPUT_FILE = "sampledKDD.csv"
	COMPAS_INPUT_FILE = "DataPreprocessing/kdd.csv"


	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])

	""" Feature normalization and one hot encoding """

	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
	y[y==0] = -1

	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	x_control = defaultdict(list)

	feature_names = []
	for attr in FEATURES_CLASSIFICATION:
		vals = data[attr]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals) # 0 mean and 1 variance  
			vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

		else: # for binary categorical variables, the label binarizer uses just one var instead of two
			lb = preprocessing.LabelBinarizer()
			lb.fit(vals)
			vals = lb.transform(vals)

		# add to sensitive features dict
		if attr in SENSITIVE_ATTRS:
			x_control[attr] = vals


		# add to learnable features
		X = np.hstack((X, vals))

		if attr in CONT_VARIABLES: # continuous feature, just append the name
			feature_names.append(attr)
		else: # categorical features
			if vals.shape[1] == 1: # binary features that passed through lib binarizer
				feature_names.append(attr)
			else:
				for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
					feature_names.append(attr + "_" + str(k))

	# convert the sensitive feature to 1-d array
	x_control = dict(x_control)
	for k in x_control.keys():
		assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()

	return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control
