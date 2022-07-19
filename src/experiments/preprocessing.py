import pandas as pd
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
from src.dataset import ImdbDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# -----------------------------------------------------------------------------------------

from nltk.corpus import stopwords
from transformers import XLNetTokenizer, XLNetModel


# # In the
# # labeled train/test sets, a negative text has a score <= 4 out of 10,
# # and a positive text has a score >= 7 out of 10. Thus text with
# # more neutral ratings are not included in the train/test sets. In the
# # unsupervised set, text of any rating are included and there are an
# # even number of text > 5 and <= 5.

# '/Users/talsagie-private/Desktop/NLP_Proj/data/train/pos' - pos loc
# '/Users/talsagie-private/Desktop/NLP_Proj/data/train/neg' - neg loc
def load_df(path_to_train):
	# # Create a list object to import positive text from train directory
	file_names = os.listdir(path_to_train)
	# # Create dataframe to store scores, labels, and texts
	train_df = pd.DataFrame(columns=['text', 'score','label'])

	for i in range(0,len(file_names)):
		ending = file_names[i].split('_')[-1]
		res = [int(i) for i in ending.split('.') if i.isdigit()]
		train_df.loc[i,'score'] = res[0]
		train_df.loc[i,'label'] = 'positive' if path_to_train.find('/pos') != -1 else 'negative'
		with open(path_to_train +'/'+ file_names[i], "r") as file:
			train_df.loc[i,'text'] = file.read()

	return train_df
			

def Concatenate(df1, df2):
	return pd.concat([df1, df2], ignore_index=True)

def cut_dup(train_df):
	print('-----------Describe------------')
	print(train_df.describe())
	# Display duplicates
	duplicated = train_df[train_df.duplicated(keep=False)]
	duplicated.sort_values("text")
	# Drop the duplicates
	df_train = pd.DataFrame.drop_duplicates(train_df, ignore_index=True)
	print('df.train shape is:',df_train.shape)
	print('-----------Describe After Drop Dup------------')
	print(df_train.describe())

	##Still a problem. There is an issue with the row with text about Sondra Locke's film. Let's investigate.
	df_to_remove = df_train[df_train['text'].str.contains("Sondra Locke stinks in this film, but then she")]
	index_to_remove = df_to_remove.index.values.astype(int)[-1] if not df_to_remove.empty  else -1
	if index_to_remove != -1 :df_train = df_train.drop(index_to_remove)
	print('df.train shape is:',df_train.shape)
	print('-----------Describe After Drop Dup------------')
	print(df_train.describe())
	df_train = df_train.drop(df_train.index[index_to_remove])
	print('-----------Describe After Drop Dup------------')
	print(df_train.describe())
	print('-----------dtypes------------')
	print(df_train.dtypes)
	return df_train


def convert_to_category(df_train):
	df_train['score'] = df_train['score'].astype(int) # convert into integers
	df_train['label'] = pd.Categorical(df_train.label) # convert into categorical


	sns.set(color_codes=True)
	sns.set(style="white", palette="muted")

	# Plot histogram of the scores
	sns.histplot(data=df_train, x="score", bins=8)
	plt.show()

	#The graph above is pratically symmetric with no values between 5 and 6 (as previously noted) which indicates 
	#that the distribuition of negatives and positive labels may be uniform. Let's check.
	
	# Print labels bar plor
	sns.catplot(x="label", kind="count", data=df_train)
	plt.show()

	#As we can verify from the graph above, the labels are not significantly 
	#unbalanced and we do not have to worry about doing under(over)sampling or creating synthetic data.

	return df_train



# Define a function to remove unecessary characters from the text
def clean_text(text):
    # text = text.lower() #lower case
    text = re.sub(r"<[^>]*>", ' ', text)
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text


def pre_processing_for_svm(df):
	lemmatizer = nltk.stem.WordNetLemmatizer()
	df['text'] = df['text'].apply(lambda word: word.lower())
	df['text'] = df['text'].apply(word_tokenize)
	eng_stopwords = stopwords.words('english') 
	df['text'] = df['text'].apply(lambda words: [word for word in words if word not in eng_stopwords])
	df['text'] = df['text'].apply(lambda words: [word for word in words if word.isalpha()])

	def lemmatize_text(text):
		text = [lemmatizer.lemmatize(word) for word in text]
		return text

	df['text'] = df['text'].apply(lemmatize_text)
	df['text'] = df['text'].apply(lambda x: str(x))

	return df


def tokens_graph(df):
	PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
	tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	token_lens = []

	# Checking the distribution of token lengths
	df['text'].apply(lambda text: token_lens.append(len(tokenizer.encode(text, max_length=512))))
	sns.set(color_codes=True)
	sns.set(style="white", palette="muted")
	sns.histplot(data=token_lens)
	plt.xlim([0, 1024])
	plt.show()

	return tokenizer


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ImdbDataset(
    text=df.text.to_numpy(),
    label=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )