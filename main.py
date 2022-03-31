########### Getting Image Indices and labels from csv, Data split ###########
from google.colab import drive #images_001 folder is stored on my GDrive
drive.mount('/content/drive')

from PIL import Image 
import pandas as pd
import numpy as np
import os #os.path.isfile will be used to check if image filepath exists

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
# from skimage import io
# import os
import matplotlib.pyplot as plt

# Initialize Dataframe
data = pd.read_csv('/content/drive/MyDrive/ChestXRay/Data_Entry_2017_v2020.csv')

def visualize_class_distribution(num_bins: int):
  """
  Prints and plots information about distribution of classes in data
  Arguments: Number of equal-width bins to show in class distribution histogram
  Returns: Nothing
  """
  print("Class counts:\n", data['Finding Labels'].astype('category').cat.codes.value_counts())
  print("Number of classes: {}\n".format(data['Finding Labels'].astype('category').cat.codes.max()))
  data['Finding Labels'].astype('category').cat.codes.hist(bins=num_bins)
  
visualize_class_distribution(num_bins=10)

# Remove rows from dataframe for which we have no downloaded images for
image_path = '/content/drive/MyDrive/ChestXRay/ChestXRay_images/'
data['Image_Path'] = image_path + data['Image Index']
data['Image_Path'] = data['Image_Path'].apply(os.path.isfile) # 
data = data[data['Image_Path'] == True] # reduces dataframe to rows we have downloaded images for

# Class distribution of data that has been downloaded -- ~200 classes represented in downloaded data vs. ~800 classes total 
visualize_class_distribution(10)
visualize_class_distribution(100)

# Split data
data_indices = np.arange(data.shape[0])
np.random.shuffle(data_indices)

num_train = int(data.shape[0]*0.6)
num_val = int(data.shape[0]*0.2)
train_indices = data_indices[0:num_train]
val_indices = data_indices[num_train:num_train+num_val]
test_indices = data_indices[num_train+num_val:]

## Preparing Image Indices and Labels of Training Set ##
# Here, we reduce the dataframe to 2 relevant columns (image names & disease labels), containing entries for the relevant data splits
#print(data["Image Index"][train_indices].reset_index().drop(['index'], axis=1))

#ii_l ~ image indices and labels
ii_l = data[["Image Index", "Finding Labels"]].reset_index().drop(['index'], axis=1)

## CONVERTING LABELS -> VECTORS ##
#currently have: labels -> lists
#each possible label = a 1 hot vector i.e. cardiomegaly = [1 0 0 0 0 0 0 0]. Take sum of one-hot-vectors in a label row as the label vector.
#i.e. if emphysema = [0 1 0 0 0 0 0 0], then a patient with cardiomegaly & emphysema has label [1 1 0 0 0 0 0 0]

#16 Classes
coded_labels = {
          "No Finding":         np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Atelectasis":        np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Cardiomegaly":       np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Effusion":           np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Infiltration":       np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
          "Mass":               np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
          "Nodule":             np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
          "Pneumonia":          np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
          "Pneumothorax":       np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
          "Consolidation":      np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
          "Edema":              np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
          "Emphysema":          np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
          "Fibrosis":           np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
          "PT":                 np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
          "Hernia":             np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
          "Pleural_Thickening": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
}


def label_to_vec(label):
  """
  Input: list of labels (type: list of strings)
  Output: One hot coded vector for each label in input (type: list of ndarrays)
  Example: label_to_vec(["Atelectasis", "Cardiomegaly"]) = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                                        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
  """
  for i in range(len(label)):
    label[i] = coded_labels[label[i]]
  return label

#PROCEDURE: convert str->list of separated strings, then list of str labels -> list of one-hot-coded np array -> label vector (sum of one-hot coded arrays)
#label_vecs = data['Finding Labels'].apply(lambda x: x.split(sep='|')).apply(lambda x: label_to_vec(x)).apply(np.sum,axis=0)
#print(label_vecs[2].shape)
#print(label_vecs)

ii_l['Finding Labels'] = ii_l['Finding Labels'].apply(lambda x: x.split(sep='|')).apply(lambda x: label_to_vec(x)).apply(np.sum,axis=0)

train_ii_l = ii_l.iloc[train_indices]
val_ii_l = ii_l.iloc[val_indices]
test_ii_l = ii_l.iloc[test_indices]

'''
test = ["Cardiomegaly", "Nodule"]
for i in range(len(test)):
  print(coded_labels[test[i]])
coded_labels[test[0]]
'''
#print(len(ii_l), train_ii_l.shape, train_ii_l)

########### Working with Images ###########          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from torchvision.transforms import ToPILImage
#from skimage import io
#from PIL import Image

class XRayDataSet(Dataset):
  """Image dataset"""

  def __init__(self, labels, image_dir, transform=None):
    self.labels = labels
    self.image_dir = image_dir
    self.transform=transform

  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_name = os.path.join(self.image_dir,
                            self.labels.iloc[idx, 0])
    
    image = torchvision.io.read_image(img_name)[0:1] # slice images, so only keep 1 channel #io.imread(img_name) 
    #image = image.to(device)
    labels = self.labels.iloc[idx,1]
    labels = np.array([labels])
    labels = labels.astype('float')
    sample = {'image': image, 'labels': labels}
    if self.transform:
      sample['image'] = self.transform(sample['image'])
    return sample

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),                                            
    torchvision.transforms.Resize(100),
    torchvision.transforms.ToTensor()
]
)

'''
train_transform = torch.nn.Sequential(
    #torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(1000)
)
'''

def check_image_loading(dataset, indices):
  '''
  Just want to compare loaded images with corresponding rows in dataframe to get an idea whether I'm loading images correctly
  '''
  for index in range(len(indices)):
    print("INDEX: ", index)
    print(train_ii_l.iloc[index], "\n")
    print(dataset[index], dataset[index]['image'].shape)
    plt.imshow(dataset[index]['image'][0,:,:])
    plt.show()

train_data = XRayDataSet(train_ii_l, image_path, train_transform)
val_data = XRayDataSet(val_ii_l, image_path)
test_data = XRayDataSet(test_ii_l, image_path)

check_image_loading(train_data, np.random.choice(len(train_ii_l), 10)) # Visualize loaded images
#check_image_loading(train_data, [2068])

train_dataloader = DataLoader(train_data, batch_size=16)
val_dataloader = DataLoader(val_data, batch_size=16)
test_dataloader = DataLoader(test_data, batch_size=16)

# want X_train.shape = (67272, height, width) # note that dataset images are black and white therefore we do not have 3 RGB channels for each image

########### TRAIN MODEL(S) ###########
baseline = models.BaseLineCNN() # re-instantiate model every time you run through a training loop
baseline.to(device)

optim = torch.optim.Adam(baseline.parameters())
loss_func = torch.nn.BCEWithLogitsLoss()#torch.nn.CrossEntropyLoss()

def loss_function(X, Y, model):
  #print("Test", Y[0], X[0])
  logits = model(X)
  Y = Y.squeeze(1)
  #Y = Y.float()
  return loss_func(logits, Y)

def predict(model, inputs):
  return torch.nn.Softmax(dim=1)(model(inputs))

def update_weights(X, Y, model):
  '''
  Redundant function - update_weights now implemented within training loop

  input: X (num examples x num labels)
         Y (num examples x num labels),
         model (nn.Module)
  '''
  loss = loss_function(X, Y, model)
  print("LOSS: ", loss)
  loss.backward()
  optim.step() # update weights

def train(num_epochs = 10, batch_size = 16):
  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  for epoch in range(num_epochs):
    #optim.zero_grad() # important !!
    avg_train_loss = 0
    avg_val_loss = 0
    #Training
    for it in range(0, 128, batch_size): # for it in range(0, len(train_ii_l) // 5, batch_size): # onlt going through small subset of data because of memory issues
      optim.zero_grad()
      print(it)
      images, labels = next(iter(train_dataloader))['image'], next(iter(train_dataloader))['labels']
      images = images / 255
      images, labels = images.to(device), labels.to(device)
      #update_weights(images, labels, baseline)

      loss = loss_function(images, labels, baseline)
      loss.backward()
      optim.step()

      train_prediction = predict(baseline, images).argmax(1)
      #avg_train_acc += train_prediction
      
      # This is most likely why I was getting the RAM issue; I did not use loss.item() before, so I was adding the entire computational graph to the loss list
      avg_train_loss += loss.detach().item() #loss_function(images, labels, baseline)

    avg_train_loss_epoch.append(avg_train_loss / train_ii_l.shape[0]) # divide by number images in batch
    #del images, labels
    #torch.cuda.synchronize()

    '''
    #Validation
    for it in range(0, 128, batch_size): # for it in range(0, len(train_ii_l) // 5, batch_size):
      print(it)
      images, labels = next(iter(val_dataloader))['image'], next(iter(val_dataloader))['labels']
      images = images / 255
      images, labels = images.to(device), labels.to(device)
      update_weights(images, labels, baseline)
      avg_val_loss += loss_function(images, labels, baseline)
    avg_val_loss_epoch.append(avg_val_loss / val_ii_l.shape[0])
    '''

  #Plot losses
  plt.title("Train vs. Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()
          
#print(torch.cuda.memory_allocated())
#torch.cuda.empty_cache()
#torch.cuda.synchronize()
train()
