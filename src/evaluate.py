from collections import defaultdict
from sklearn import metrics
import torch
import numpy as np
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from src.model.main_model import train_epoch

EPOCHS = 3





def hyperparameters(model,train_data_loader):
	EPOCHS = 3

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
									{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
									{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

	total_steps = len(train_data_loader) * EPOCHS

	scheduler = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=0,
	num_training_steps=total_steps
	)

	return optimizer, scheduler

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
  
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(4,512).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(targets, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


##fine tuning

def fine_tuning(model, optimizer, device, scheduler,train_data_loader,df_train,val_data_loader,df_val):
	history = defaultdict(list)
	best_accuracy = 0

	for epoch in range(EPOCHS):
		print(f'Epoch {epoch + 1}/{EPOCHS}')
		print('-' * 10)

		train_acc, train_loss = train_epoch(
			model,
			train_data_loader,     
			optimizer, 
			device, 
			scheduler, 
			len(df_train)
		)

		print(f'Train loss {train_loss} Train accuracy {train_acc}')

		val_acc, val_loss = eval_model(
			model,
			val_data_loader, 
			device, 
			len(df_val)
		)

		print(f'Val loss {val_loss} Val accuracy {val_acc}')
		print()

		history['train_acc'].append(train_acc)
		history['train_loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		if val_acc > best_accuracy:
			torch.save(model.state_dict(), '/content/drive/My Drive/NLP/Sentiment Analysis Series/models/xlnet_model.bin')
			best_accuracy = val_acc

##Evaluation of the fine-tuned model
def eval(model,test_data_loader,device,df_test):
	model.load_state_dict(torch.load('/content/drive/My Drive/NLP/Sentiment Analysis Series/models/xlnet_model.bin'))
	model = model.to(device)
	test_acc, test_loss = eval_model(
	model,
	test_data_loader,
	device,
	len(df_test)
	)

	print('Test Accuracy :', test_acc)
	print('Test Loss :', test_loss)



def get_predictions(model, data_loader,device):
    model = model.eval()
    
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["review_text"]
            input_ids = d["input_ids"].reshape(4,512).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)

            loss = outputs[0]
            logits = outputs[1]
            
            _, preds = torch.max(outputs[1], dim=1)

            probs = F.softmax(outputs[1], dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values