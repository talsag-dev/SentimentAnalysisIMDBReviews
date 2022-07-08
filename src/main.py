# The main flow should be here
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import  accuracy_score as acuuracy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import XLNetForSequenceClassification
import torch
#-------------------------------------
from src.model.main_model import test_train_split
from src.model.tfidf import tfidfHandle
from src.model.naive_bayes_mode import naive_classifer,NB_model_with_cv
from src.model.svm_model import svm_model,svm_model_with_cv
from src.experiments.preprocessing import create_data_loader,tokens_graph,load_df, Concatenate,cut_dup,convert_to_category,clean_text,pre_processing_for_svm
from src.evaluate import hyperparameters,fine_tuning,get_predictions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load train df
train_df_pos = load_df('/Users/talsagie-private/Desktop/NLP Proj/data/train/pos')
train_df_neg = load_df('/Users/talsagie-private/Desktop/NLP Proj/data/train/neg')
# Concatenate pos and neg train dfs
train_df = Concatenate(train_df_pos, train_df_neg)

# load train df - Watch for the head of df
print('-----------Train DF------------')
print(train_df.head())


#load test df
test_df_pos = load_df('/Users/talsagie-private/Desktop/NLP Proj/data/test/pos')
test_df_neg = load_df('/Users/talsagie-private/Desktop/NLP Proj/data/test/neg')
# Concatenate pos and neg test dfs
test_df = Concatenate(test_df_pos, test_df_neg)
# load test df - Watch for the head of df
print('-----------Test DF------------')
print(test_df.head())

print('-----------Train DF------------')
print(train_df.head())


# Preprocess train df
df_train = cut_dup(train_df)
df_train=convert_to_category(df_train)


# Preprocess test df
df_test = cut_dup(test_df)
df_test=convert_to_category(df_test)

# Apply the clean_text function of both data sets
df_train['text'] = df_train['text'].apply(clean_text)
df_test['text'] = df_test['text'].apply(clean_text)
print('-----------Train DF------------')
print(df_train.head()) # display the first 5 rows of Train subset

# Replace labels with numbers on both subsets
df_train['label'] = df_train['label'].map({'positive':1, 'negative':0})
df_test['label'] = df_test['label'].map({'positive':1, 'negative':0})
print('-----------Test DF------------')
print(df_test.head()) # display the first 5 rows of Test subset


# pre-process for svm and nb
#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# df_train_baseline = pre_processing_for_svm(df_train)
# df_test_baseline = pre_processing_for_svm(df_test)
#-------------------------------------------------------------------------------------------

# Shuffle rows
#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# df_train_baseline = shuffle(df_train_baseline)
# df_test_baseline = shuffle(df_test_baseline)

# print('-----------Train DF------------')
# print(df_train_baseline.head())
# print('-----------Test DF------------')
# print(df_test_baseline.head())
#-------------------------------------------------------------------------------------------

# Split Train and Test subsets , we didnt use split_test_train function 
# because we want to keep the same test and train subsets as its already divided

#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# X_train, y_train = test_train_split(df_train_baseline)
# X_test, y_test = test_train_split(df_test_baseline)
# -----------------------------------------------------------------------------------------

## Baseline -SVM and NB:


##NB
#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# Train_X_Tfidf,Test_X_Tfidf = tfidfHandle(X_train,X_test,Concatenate(df_train_baseline,df_test_baseline))
# naive_classifer(Train_X_Tfidf,y_train,Test_X_Tfidf,y_test)
# -----------------------------------------------------------------------------------------

## with navie bayse  we get acc of 83%


##SVM
#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# svm_model(Train_X_Tfidf,y_train,Test_X_Tfidf,y_test)
# -----------------------------------------------------------------------------------------
##with svm we get acc of 87.5%



# lets try and use cross-validation:

#----------------- TO BE UNCOMMENTED WHEN WE HAVE THE MODEL -------------------------------
# NB_model_with_cv(Train_X_Tfidf,y_train)
# svm_model_with_cv(Train_X_Tfidf,y_train)
# -----------------------------------------------------------------------------------------
##with cross validation we get acc of 87% for NB and 89%



# Models based on Transformers like XLNet consume lots of computational resources. Even 
# using a GPU it would take a significant
#  amount of time to train and validate the model.
#  That is why we decided to reduce the number of rows.


# # Reduce number of rows
df_train = df_train[:12000]
df_test = df_test[:12000]

df_train = shuffle(df_train)
df_test = shuffle(df_test)

df = Concatenate(df_train, df_test)

tokenizer = tokens_graph(df)

## therefore MAX_LEN = 512

df_train, df_test = train_test_split(df, test_size=0.5, random_state=101)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=101)

MAX_LEN = 512
BATCH_SIZE = 4

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
model = model.to(device)
print(model)


optimizer, scheduler = hyperparameters(model,train_data_loader)
fine_tuning(model, optimizer, device, scheduler,train_data_loader,df_train,val_data_loader,df_val)
model.load_state_dict(torch.load('/content/drive/My Drive/NLP/Sentiment Analysis Series/models/xlnet_model.bin'))
model = model.to(device)
eval(model,test_data_loader, device,  df_test)


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader,
  device
)




print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
