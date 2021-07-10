'''
Prediction of credit card customers that are most likely to churn

author: Julia
date: July 10, 2021
'''


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
	'''
	returns dataframe for the csv found at pth

	input:
			pth: a path to the csv
	output:
			df: pandas dataframe
	'''
	df = pd.read_csv(pth)
	return df


def perform_eda(df):
	'''
	perform eda on df and save figures to images folder
	input:
			df: pandas dataframe

	output:
			None
	'''

	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

	plt.figure(figsize=(20,10)) 
	df['Churn'].hist();
	plt.savefig('./images/eda/churn_distribution.png')   # save the figure to file
	plt.close()    # close the figure window

	plt.figure(figsize=(20,10)) 
	df['Customer_Age'].hist();
	plt.savefig('./images/eda/customer_age_distribution.png')

	plt.figure(figsize=(20,10)) 
	df.Marital_Status.value_counts('normalize').plot(kind='bar');
	plt.savefig('./images/eda/martial_status_distribution.png')
	plt.close() 

	plt.figure(figsize=(20,10)) 
	sns.distplot(df['Total_Trans_Ct']);
	plt.savefig('./images/eda/total_transaction_distribution.png')
	plt.close() 

	plt.figure(figsize=(20,10)) 
	sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
	plt.savefig('./images/eda/heatmap.png')
	plt.close() 

	return df


def encoder_helper(df, category_lst):
	'''
	helper function to turn each categorical column into a new column with
	propotion of churn for each category - associated with cell 15 from the notebook

	input:
			df: pandas dataframe
			category_lst: list of columns that contain categorical features

	output:
			df: pandas dataframe with new columns for
	'''
	for category in category_lst:
		tmp = []
		category_groups = df.groupby(category).mean()['Churn']

		for val in df[category]:
			tmp.append(category_groups.loc[val])

		df[category + '_Churn'] = tmp
	
	return df


def perform_feature_engineering(df):
	'''
	input:
			  df: pandas dataframe
			  
	output:
			  X_train: X training data
			  X_test: X testing data
			  y_train: y training data
			  y_test: y testing data
	'''
	keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
				'Total_Relationship_Count', 'Months_Inactive_12_mon',
				'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
				'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
				'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
				'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
				'Income_Category_Churn', 'Card_Category_Churn']

	X = pd.DataFrame()
	X[keep_cols] = df[keep_cols]
	y = df['Churn']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

	return X_train, X_test, y_train, y_test




def classification_report_image(y_train,
								y_test,
								y_train_preds_lr,
								y_train_preds_rf,
								y_test_preds_lr,
								y_test_preds_rf):
	'''
	produces classification report for training and testing results and stores report as image
	in images folder
	input:
			y_train: training response values
			y_test:  test response values
			y_train_preds_lr: training predictions from logistic regression
			y_train_preds_rf: training predictions from random forest
			y_test_preds_lr: test predictions from logistic regression
			y_test_preds_rf: test predictions from random forest

	output:
			 None
	'''
	
	

	plt.rc('figure', figsize=(7, 5))
	plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
	plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
	plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
	plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
	plt.axis('off');
	plt.savefig('./images/results/logistics_results.png')
	plt.close() 

	plt.rc('figure', figsize=(7, 5))
	plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
	plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
	plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
	plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
	plt.axis('off');
	plt.savefig('./images/results/rf_results.png')
	plt.close() 


def feature_importance_plot(model, X_data, output_pth):
	'''
	creates and stores the feature importances in pth
	input:
			model: model object containing feature_importances_
			X_data: pandas dataframe of X values
			output_pth: path to store the figure

	output:
			 None
	'''

	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	names = [X_data.columns[i] for i in indices]
	
	plt.figure(figsize=(20,5))
	plt.title("Feature Importance")
	plt.ylabel('Importance')
	plt.bar(range(X_data.shape[1]), importances[indices])
	plt.xticks(range(X_data.shape[1]), names, rotation=90);

	plt.savefig(output_pth + 'feature_importance.png')
	plt.close()


def train_models(X_train, X_test, y_train, y_test, do_train):
	'''
	train, store model results: images + scores, and store models
	input:
			  X_train: X training data
			  X_test: X testing data
			  y_train: y training data
			  y_test: y testing data
			  do_train: flag for train model
	output:
			  None
	'''
	if do_train==1:
		
		rfc = RandomForestClassifier(random_state=42)
		lrc = LogisticRegression()

		param_grid = { 
			'n_estimators': [200, 500],
			'max_features': ['auto', 'sqrt'],
			'max_depth' : [4,5,100],
			'criterion' :['gini', 'entropy']
		}

		cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
		cv_rfc.fit(X_train, y_train)

		lrc.fit(X_train, y_train)

		# save best model
		joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
		joblib.dump(lrc, './models/logistic_model.pkl')

	else: 
		rfc_model = joblib.load('./models/rfc_model.pkl')
		lr_model = joblib.load('./models/logistic_model.pkl')


	# predict customer churn
	y_train_preds_rf = rfc_model.predict(X_train)
	y_test_preds_rf = rfc_model.predict(X_test)


	y_train_preds_lr = lr_model.predict(X_train)
	y_test_preds_lr = lr_model.predict(X_test)

	# save scores
	classification_report_image(y_train,
								y_test,
								y_train_preds_lr,
								y_train_preds_rf,
								y_test_preds_lr,
								y_test_preds_rf)


	# save feature importance 

	feature_importance_plot(rfc_model, X_train, './images/results/')

	


if __name__ == "__main__":
	
	pth = "./data/bank_data.csv"
	category_lst = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
	
	df = import_data(pth)
	df = perform_eda(df)
	df = encoder_helper(df, category_lst)
	X_train, X_test, y_train, y_test = perform_feature_engineering(df)
	train_models(X_train, X_test, y_train, y_test, 0)

	
	
