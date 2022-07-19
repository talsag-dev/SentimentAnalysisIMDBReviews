import tensorflow as tf
from sklearn import metrics
from torch import nn, optim
import torch
import numpy as np

def test_train_split(df_train):
	# Get Train and Test features and labels
	X_train = df_train.text; y_train = df_train.label

	# Transform from series into lists/arrays
	X_train = X_train.tolist()
	y_train = y_train.to_numpy()
	# Check size of Train and Test subsets
	print('input size: ', len(X_train))
	print('output size: ', len(y_train))
	return X_train, y_train

	

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    acc = 0
    counter = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].reshape(4,512).to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["label"].to(device)
        
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = label)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        label = label.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(label, prediction)

        acc += accuracy
        losses.append(loss.item())
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)