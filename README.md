# Diagnoses of Chest X-Rays with DCNN(s)
This project aims to use Deep Convolutional Neural Network(s) to read and
diagnose Chest X-ray images in accordance with a set of 15 common thoracic
diseases.

## Background
The [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) used for
*multilabel* classification is openly provided by the NIH clinical center. It
takes up ~46GB of storage space, comprising ~112,000 Chest X-ray images, each
of which have corresponding labels available in the above link. Each image can
be indicative of any combination of the below diseases:

- No Finding
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- PT
- Hernia
- Pleural Thickening

## File Structure
**main.py**: Preprocesses provided csv to generate one-hot encodings for image
labels, splits dataset, instantiates and trains DCNN architectures available
from *model.py* file.

**models.py**: Implements DCNN architecture(s) used in *main.py*.

**resize__image.py**: This project is developed using Google Colab/Drive, which
have a 15GB storage limit. To deal with this, the provided images are reduced
in size so that it becomes feasible to work with the entire dataset. A
limitation however, is that the method of resizing currently used may introduce
distortions that negatively impact the accuracy of the DCNN(s) used.

## Requirements
- Google Colab (GPU)
- PyTorch
- Pandas
- NumPy
- Matplotlib.pyplot
- 45.6GB of local storage space

## How to use this
1. Download the [image dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737).
2. Create a new file named "Resized_Images" in your project directory, and run *resize__image.py*
3. Upload the folder of resized images to Google Drive. This may take several
   hours.
4. Run the contents of *main.py* and *models.py* from Google Colab. You may
   have to change the file paths used in *main.py* to suit your Google Drive
file structure.
