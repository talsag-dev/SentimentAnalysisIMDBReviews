from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score



def svm_model(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y):
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(Train_X_Tfidf,Train_Y)
	# predict the labels on validation dataset
	predictions_SVM = SVM.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


def svm_model_with_cv(Train_X_Tfidf,Train_Y):
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	scores = cross_val_score(SVM, Train_X_Tfidf, Train_Y, cv=5)
	# predict the labels on validation dataset
	print('SVM scores with CrossVal -> ',scores)

	print("SVM model with CrossVal %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

