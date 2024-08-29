import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

# Dataset class

class ABDataset(Dataset):
  '''
    Dataset class to hold AcousticBrainz features and labels.
  '''
  def __init__(self, data, labels):
    self.labels = labels
    self.data = data

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

# Neural Network Model

class ABModel(nn.Module) :
  '''
    Neural Network for training on Acoustic Brainz data.
    There is always 4 outputs, corresponding to the subgenre classes.
    The size of the input layer and hidden layers are specified in the constructor.
  '''
  def __init__(self, *hidden_layer_size, mu=0.0, sigma=1.0) :
    super().__init__()
    self.mean = mu
    self.std = sigma
    self.model = nn.Sequential(
      *[nn.Sequential(
          nn.Linear(hidden_layer_size[i], hidden_layer_size[i+1]),
          nn.ReLU())
       for i in range(0, len(hidden_layer_size)-1)],
      nn.Linear(hidden_layer_size[-1], 4),
      nn.Sigmoid()
    ).to(device="cuda:0")

  def forward(self, input) :
    return self.model((input-self.mean)/self.std)

# Model Training

def train_one_epoch(p_model, p_loss_fn, p_optimizer, p_training_loader):
  '''
    Trains one epoch of p_model.
  '''
  running_loss = 0.
  last_loss = 0.
  
  for i, data in enumerate(p_training_loader):
      
    inputs, labels = data
    p_optimizer.zero_grad()
    outputs = p_model(inputs)
    loss = p_loss_fn(outputs, labels)
    loss.backward()
    p_optimizer.step()
    running_loss += loss.item()
    if i % 1000 == 999:
      last_loss = running_loss / 1000

  return last_loss
  
def train(num_epochs, p_model, p_loss_fn, p_optimizer, p_training_loader, print_scores=False) :
  '''
    Trains a model for the given number of epochs.
  '''
  for i in range(0, num_epochs) :
	  
    if print_scores :
        print("BCELoss for {}th epoch : {}".format(i+1, \
            train_one_epoch(p_model, p_loss_fn, p_optimizer, p_training_loader)))
    else :
      train_one_epoch(p_model, p_loss_fn, p_optimizer, p_training_loader)

# Model scoring

def score(model, validation_loader) :
  '''
    Returns the accuracy, recall, precision for each class of a model evaluated on validation_loader.
  '''
  with torch.no_grad() :
    
    true_positive = torch.tensor([0,0,0,0]).to(device="cuda:0")
    false_negative = torch.tensor([0,0,0,0]).to(device="cuda:0")
    false_positive = torch.tensor([0,0,0,0]).to(device="cuda:0")
    true_total = torch.tensor([0,0,0,0]).to(device="cuda:0")
    total = 0
    
    for x, t, in validation_loader:
        
      prediction = (model(x) > 0.5).squeeze(0)
      true = (prediction==t.squeeze(0))
      total += 1
      true_positive += torch.logical_and(prediction, true)
      true_total += true
      false_positive += torch.logical_and(prediction, torch.logical_not(true))
      false_negative += torch.logical_and(torch.logical_not(prediction), torch.logical_not(true))
    return {"Accuracy" : true_total/total,
            "Recall" : true_positive/(true_positive+false_positive),
            "Precision" : true_positive/(true_positive+false_negative)}

def print_scores(model, validation_trainer) :
  '''
    Prints model scores evaluated on validation_trainer.
  '''
  d = score(model, validation_trainer)
  genres = ["Trance", "House", "Techno", "Drum and Bass"]
  print("\t", f"{'' : <{10}} {genres[0] : <{10}} {genres[1] : <{10}} {genres[2] : <{10}} {genres[3] : <{15}}")
    
  for s in d.keys() :
      
    t = torch.round(d[s], decimals=4)
    print("\t", f"{s:<10} {t[0]:<10.4f} {t[1]:<10.4f} {t[2]:<10.4f} {t[3]:<10.4f}")

def plot_scores(scores, title) :
  '''
    Plots scores dictionary into a bar plot.
  '''
  classes = ['Trance', 'House', 'Techno', 'Drum and Bass']
  scoring = ("Accuracy", "Recall", "Precision")
  data = {
    k : tuple(scores[s].cpu().numpy()[i] for s in scoring)
    for i, k in enumerate(classes)
  }

  x = np.arange(len(scoring))  # the label locations
  width = 0.22  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in data.items():
      
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(measurement, 3), width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Score')
  ax.set_title(title)
  ax.legend(loc='best', ncols=2)

  ax.set_xticks(x + width, scoring)
  ax.set_ylim(0, 1)

  plt.show()