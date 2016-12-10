# log_reg.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class Log_Reg:

	def __init__(self):
		self.lg = LogisticRegression(penalty='l2', C=1, fit_intercept=True, intercept_scaling=1000)

	def train_test(self, train_data, train_labels, test_data, test_lables):
		self.lg.fit(train_data, train_labels)
		scores = self.lg.score(test_data, test_lables)
		prd = self.lg.predict_proba(test_data)[:,1]
		for p,lab in zip(prd,test_lables):
			print("%.3f\t"%(p,) + str(lab))
		# probs = self.lg.predict_proba(test_data)
		print("lg: %.3f%%" % (scores*100, ))

	def kfold_log_reg(self, data, labels, files):
		seed=7
		np.random.seed(seed)

		cvscores = []
		f1scores = []
		prscores = []
		rcscores = []
		tnlist = []
		tplist = []
		fnlist = []
		fplist = []
		lentest = []
		lentrain = []
		lentotal = []
		pearson = []

		kfold = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=seed)

		for train, test in kfold:
			train_data = data[train]
			test_data = data[test]
			train_labels = labels[train]
			test_lables = labels[test]
			test_files = [files[i] for i in test]
			self.lg.fit(train_data, train_labels)

			# evaluate the model
			scores = self.lg.score(test_data, test_lables)
			prd = self.lg.predict(test_data)
			probs = self.lg.predict_proba(test_data)
			print("lg: %.2f%%" % (scores*100, ))
			cvscores.append(scores * 100)
			#tst = np.array([item[0] for item in data[test]])
			#pearson.append(pearsonr(tst.ravel(), labels[test].ravel()))

			bias_prd = []
			bias_labels = []
			nop = 0

			for i,x in enumerate(probs):
				if x[0] >= .5:
					bias_prd.append(0.0)
					bias_labels.append(test_lables[i])
				elif x[1] >= .5:
					bias_prd.append(1.0)
					bias_labels.append(test_lables[i])
				else:
					nop+=1

			# for i,x in enumerate(prd):
			# 	if x==0 and test_lables[i]==1:
			# 		print(prd[i])
			# 		print(probs[i])
			# 		print(test_files[i])
			# 		print(data[test[i]])

			f1scores.append(f1_score(test_lables, prd, average="binary"))
			prscores.append(precision_score(test_lables, prd, average="binary"))
			rcscores.append(recall_score(test_lables, prd, average="binary"))
			tn, fp, fn, tp = confusion_matrix(test_lables, prd).ravel()
			tnlist.append(tn)
			tplist.append(tp)
			fnlist.append(fn)
			fplist.append(fp)
			lentest.append(len(test))
			lentrain.append(len(train))
			lentotal.append(len(labels))
			tn, fp, fn, tp = confusion_matrix(bias_labels, bias_prd).ravel()
			print((tp, tn, fp, fn, nop, float(nop)/len(test_lables)))

		print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
		print("F1: %.2f (+/- %.2f)" % (np.mean(f1scores), np.std(f1scores)))
		print("Precision: %.2f (+/- %.2f)" % (np.mean(prscores), np.std(prscores)))
		print("Recall: %.2f (+/- %.2f)" % (np.mean(rcscores), np.std(rcscores)))
		print("TN: %.2f (+/- %.2f)" % (np.mean(tnlist), np.std(tnlist)))
		print("TP: %.2f (+/- %.2f)" % (np.mean(tplist), np.std(tplist)))
		print("FN: %.2f (+/- %.2f)" % (np.mean(fnlist), np.std(fnlist)))
		print("FP: %.2f (+/- %.2f)" % (np.mean(fplist), np.std(fplist)))
		#print("Pearson correlation: %.2f (+/- %.2f)" % (np.mean(pearson), np.std(pearson)))
		print("TOTAL TEST: %.2f (+/- %.2f)" % (np.mean(lentest), np.std(lentest)))
		print("TOTAL TRAIN: %.2f (+/- %.2f)" % (np.mean(lentrain), np.std(lentrain)))
		print("TOTAL: %.2f (+/- %.2f)" % (np.mean(lentotal), np.std(lentotal)))
		print(fnlist)
