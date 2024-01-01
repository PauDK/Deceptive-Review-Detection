import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from torchmetrics import Accuracy
import tqdm.notebook


class BertWithClassifier(nn.Module):
    '''
    Bert model with classification head
    '''
    def __init__(self, 
                 bert_model, 
                 max_seq_length=512, 
                 loss_fn=nn.CrossEntropyLoss):
        '''
        Args:
            bert_model: Huggingface BertModel object
            max_seq_length: Max length of input text sequence
            loss_fn: loss function
        '''
        super(BertWithClassifier, self).__init__()
        self.max_seq_length = max_seq_length
        self.bert_layer = bert_model
        self.fc = nn.Linear(768, 2)
        self.loss = loss_fn()
    
    def forward(self, input_ids, token_type_ids, attention_mask, y, train=True):
        '''
        Forward pass of the model
        Args:
            input_ids (torch.tensor [batch_size, max_seq_length, hidden_size]): input tokens
            token_type_ids (torch.tensor [batch_size, max_seq_length]): type of token
            attention_mask (torch.tensor [batch_size, max_seq_length]): mask indicator
            y (torch.tensor [batch_size]): ground truth labels
            train (bool): boolean indicating training time
        Returns:
            pred (torch.tensor [batch_size, num_class]): normalize score for each class
            loss (torch.tensor [1]): loss
        '''
        if not train:
            with torch.no_grad():
                bert_output = self.bert_layer(input_ids, token_type_ids, attention_mask)
                unnormalized_score = self.fc(bert_output.pooler_output)
                pred = softmax(unnormalized_score, dim=-1)
                loss = self.loss(unnormalized_score, y)
                return pred, loss
    
        bert_output = self.bert_layer(input_ids, token_type_ids, attention_mask)
        unnormalized_score = self.fc(bert_output.pooler_output)
        pred = softmax(unnormalized_score, dim=-1)
        loss = self.loss(unnormalized_score, y)
        return pred, loss


class dataset(Dataset):
    '''
    Dataset object inheritted from torch.utils.data.Dataset to facilitate training
    '''
    def __init__(self, tokenizer, X, y, max_seq_length=512):
        embeddings = tokenizer(X, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        self.input_ids = embeddings.input_ids
        self.token_type_ids = embeddings.token_type_ids
        self.attention_mask = embeddings.attention_mask
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.input_ids[idx]
        item['token_type_ids'] = self.token_type_ids[idx]
        item['attention_mask'] = self.attention_mask[idx]
        item_label = self.labels[idx]
        return item, item_label


def train(model, 
          data_loader, 
          val_data_loader, 
          optimizer,
          learning_rate,  
          num_epochs,
          device,
          batch_size,
          data_size,
          scheduler=None):
    opt = optimizer(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-07)
    if scheduler is not None:
        steps_per_epoch = data_size // batch_size
        num_train_steps = steps_per_epoch * num_epochs
        num_warmup_steps = num_train_steps // 10
        sch = scheduler(opt, num_warmup_steps, num_train_steps)
    train_losses = []
    val_losses = []
    for epoch in tqdm.notebook.trange(num_epochs, desc='training', unit='epoch'):
        with tqdm.notebook.tqdm(data_loader,
                                desc='epoch {}'.format(epoch + 1),
                                unit='batch',
                                total=len(data_loader)
                               ) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch_data in enumerate(batch_iterator, start=1):
                opt.zero_grad()
                input_ids, token_type_ids, attention_mask = (val.to(device) for val in batch_data[0].values())
                train_targets = batch_data[1].to(device)
                pred, loss = model.forward(input_ids, token_type_ids, attention_mask, train_targets, train=True)
                total_loss += loss
                loss.backward()
                opt.step()
                batch_iterator.set_postfix(mean_loss = total_loss / i, current_loss=loss.item())
                sch.step()
            train_losses.append(total_loss.unsqueeze(0) / i)
            val_preds, val_loss, val_acc = validate(model, val_data_loader, device)
            print(val_acc)
            val_losses.append(val_loss.unsqueeze(0))
    return train_losses, val_losses


def validate(model, data_loader, device):
    total_loss = 0.0
    accuracy = Accuracy().to(device)
    preds = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    for i, batch in enumerate(data_loader):
        input_ids, token_type_ids, attention_mask = (val.to(device) for val in batch[0].values())
        batch_targets = batch[1].to(device)
        pred, loss = model.forward(input_ids, token_type_ids, attention_mask, batch_targets, train=False)
        total_loss += loss
        preds = torch.cat([preds, pred])
        targets = torch.cat([targets, batch_targets])
    total_loss /= i
    targets = targets.int().to(device)
    acc = accuracy(preds, targets)
    return preds, total_loss, acc
