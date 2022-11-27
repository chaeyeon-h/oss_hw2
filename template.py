#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/chaeyeon-h/oss_hw2.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn

def load_dataset(dataset_path): 
	data = pd.read_csv(dataset_path)
	return data

def dataset_stat(dataset_df):
	data=dataset_df
	countZero=0
	countOne=0
	for i in range(0,len(data)):
		if data['target'][i]==0:
			countZero+=1
		else:
			countOne+=1
	numFeature=len(data.columns)-1
	return numFeature,countZero,countOne

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	data=dataset_df.to_numpy()
	data_input=data[:,:-1]
	data_target=data[:,-1]
	x_train, x_test,y_train, y_test = train_test_split(data_input, data_target, test_size=testset_size, random_state=0)
	return x_train, x_test, y_train, y_test	

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt=DecisionTreeClassifier(random_state=42)
	dt.fit(x_train,y_train)
	
	y_predicted = dt.predict(x_test)

	accuracy=sklearn.metrics.accuracy_score(y_test,y_predicted)
	precision=sklearn.metrics.precision_score(y_test,y_predicted)
	recall=sklearn.metrics.recall_score(y_test,y_predicted)

	return accuracy,precision,recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf=RandomForestClassifier(random_state=42)
	rf.fit(x_train,y_train)
	y_predicted=rf.predict(x_test)

	accuracy=sklearn.metrics.accuracy_score(y_test,y_predicted)
	precision=sklearn.metrics.precision_score(y_test,y_predicted)
	recall=sklearn.metrics.recall_score(y_test,y_predicted)

	return accuracy,precision,recall	


def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this 
	svc_pipeline=make_pipeline(
		StandardScaler()
		,SVC())
	svc_pipeline.fit(x_train,y_train)
	y_predicted=svc_pipeline.predict(x_test)

	accuracy=sklearn.metrics.accuracy_score(y_test,y_predicted)
	precision=sklearn.metrics.precision_score(y_test,y_predicted)
	recall=sklearn.metrics.recall_score(y_test,y_predicted)

	return accuracy,precision,recall	


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)