from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def naive_classifer(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y):
	# fit the training dataset on the NB classifier
	Naive = naive_bayes.MultinomialNB()
	Naive.fit(Train_X_Tfidf,Train_Y)
	# predict the labels on validation dataset
	predictions_NB = Naive.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)



def NB_model_with_cv(Train_X_Tfidf,Train_Y):
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	Naive = naive_bayes.MultinomialNB()
	Naive.fit(Train_X_Tfidf,Train_Y)
	scores = cross_val_score(Naive, Train_X_Tfidf, Train_Y, cv=5)
	# predict the labels on validation dataset
	print('NB scores with CrossVal ->',scores)

	print("NB cross val %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

