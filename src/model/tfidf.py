from sklearn.feature_extraction.text import TfidfVectorizer



def tfidfHandle(Train_X,Test_X,df):
	Tfidf_vect = TfidfVectorizer(max_df=0.7)
	Tfidf_vect.fit(df['text'])
	Train_X_Tfidf = Tfidf_vect.transform(Train_X)
	Test_X_Tfidf = Tfidf_vect.transform(Test_X)

	return Train_X_Tfidf,Test_X_Tfidf