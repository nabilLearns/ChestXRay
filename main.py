from google.colab import drive #images_001 folder is stored on my GDrive
drive.mount('/content/drive')

from PIL import Image 
import pandas as pd
import numpy as np
import os #os.path.isfile will be used to check if image filepath exists

# Initialize Dataframe
data = pd.read_csv('/content/drive/MyDrive/ChestXRay/Data_Entry_2017_v2020.csv')

# Remove rows from dataframe for which we have no downloaded images for
image_path = image_path = '/content/drive/MyDrive/ChestXRay/ChestXRay_images/'
data['Image_Path'] = image_path + data['Image Index']
data['Image_Path'] = data['Image_Path'].apply(os.path.isfile) # 
data = data[data['Image_Path'] == True] # reduces dataframe to rows we have downloaded images for

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


print(ii_l.info(), ii_l)

'''
train_ii_l = data[["Image Index", "Finding Labels"]].iloc[train_indices].reset_index().drop(['index'], axis=1)
val_ii_l = data[["Image Index", "Finding Labels"]].iloc[val_indices].reset_index().drop(['index'], axis=1)
test_ii_l = data[["Image Index", "Finding Labels"]].iloc[test_indices].reset_index().drop(['index'], axis=1)
#print(train_ii_l)
'''

## CONVERTING LABELS -> VECTORS ##
#currently have: labels -> lists
#each possible label = a 1 hot vector i.e. cardiomegaly = [1 0 0 0 0 0 0 0]. Take sum of one-hot-vectors in a label row as the label vector.
#i.e. if emphysema = [0 1 0 0 0 0 0 0], then a patient with cardiomegaly & emphysema has label [1 1 0 0 0 0 0 0]

#15 Classes
coded_labels = {
          "No Finding":         np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Atelectasis":        np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Cardiomegaly":       np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Effusion":           np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
          "Infiltration":       np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
          "Mass":               np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
          "Nodule":             np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
          "Pneumonia":          np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
          "Pneumothorax":       np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
          "Consolidation":      np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
          "Edema":              np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
          "Emphysema":          np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
          "Fibrosis":           np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
          "PT":                 np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
          "Hernia":             np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
          "Pleural_Thickening": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
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
train_ii_l['Finding Labels'] = train_ii_l['Finding Labels'].apply(lambda x: x.split(sep='|')).apply(lambda x: label_to_vec(x)).apply(np.sum,axis=0)
val_ii_l['Finding Labels'] = val_ii_l['Finding Labels'].apply(lambda x: x.split(sep='|')).apply(lambda x: label_to_vec(x)).apply(np.sum,axis=0)
test_ii_l['Finding Labels'] = test_ii_l['Finding Labels'].apply(lambda x: x.split(sep='|')).apply(lambda x: label_to_vec(x)).apply(np.sum,axis=0)

test = ["Cardiomegaly", "Nodule"]
for i in range(len(test)):
  print(coded_labels[test[i]])
coded_labels[test[0]]
'''

print(len(ii_l), train_ii_l.shape, train_ii_l)
