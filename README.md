# SemanticSegmentationCityscapes
Deep Learning Large Project

## Quick Start
```
streamlit run dashboard.py
```

Do make sure that streamlit is downloaded (`pip install streamlit` should typically be sufficient). This will start the dashboard to interact with the data visualisation. 

In the GUI, the user can use the model and image dropdown to select the model to use for visualisation and the image to input. On the left, it visualises the Image sequence that is input into the model. Under ‘Annotated Image’, there is a colourized comparison between the ground truth and the output of the model, along with the IOU score. 

Scrolling down the page, we can see interactive charts from the training process of the loss and IOU across epochs. 

To add more models or samples to be visualised in the GUI, the user needs to add the appropriate files in ./models_display and ./data_display respectively. 


## Dataset
Note that the dataset file structure should be as follows with only the city ulm as the only example. 
```
./data
├── gtFine
│   ├── test
│   ├── train
│   │   └── ulm
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   │   └── ulm
│   └── val
└── leftImg8bit_sequence
    ├── test
    ├── train
    │   └── ulm
    └── val
```
This is important for preversing the training and evaluation process of the model. 

## Data Augmentation
### preprocessing > augment_script.py
#### augment_image
Spplies gamma correction and saturation correction to generate new image for data augmentation. This generated image would be added to the dataset for training,validation and testing in a seperate function.
Mandatory args:
1. path: file path to image
2. gamma: accepts a random value of uniform distribution between 0.5 and 1.5 to do gamma correction
3. saturation: accepts a random value of uniform distributionbetween 0.5 and 1.5 to do saturation correction
### datasets > cityscapes.py
#### setupDatasetsAndLoaders
Reduces each data sample size to 224 by 448 pixels from 1024 by 2048 pixels to speed up the processing of the dataset. 
Data normalisation is performed for each data sample by calculating the mean and standard deviation of the dataset and applying normalisation transformation.
Mandatory args:
1. batch_size: defaults to 64


## Models
There are 2 models we have explored in the project.
1. Resnet_LSTM
2. PSP
Take a look at the report to find out more about our findings.

## main.ipynb
To generate results for grid search, run all cells in this jupyter notebook. The notebook calls hyperparams_train from train.py file which selects the parameters that gives the best loss rate and accuracy for 10 epochs.

## train.py
### train
Optional args:
1. batch_size: defaults to 1
2. num_epochs: defaults to 10
3. lr: defaults to 0.001
4. optimizer_name: defaults to "adam"
5. use_psp: selects if pyramid scheme parsing is used, defaults to false

### hyperparams_train
used for grid search, iterates over set of batch sizes and learning rates for the model to determine which parameters optimise loss rate and accuracy of the model.
Optional args:
1. optimizer: defaults to "adam"
2. use_psp: selects if pyramid scheme parsing is used, defaults to true


## Other Features
### Save Checkpoint
After every epoch of training, we would save the trained model’s learned parameters. In the event that the kernel disconnects, we would be able to reload the learned parameters onto the model and continue training from the epoch that we left off.

### Predict Image
After retrieving the tensor containing that assigns the probability of one pixel being in each of the nineteen labelled classes and one unlabelled class, we select the class in which the pixel is most likely to belong to. It converts the class of the pixel into its corresponding colour. If the pixel belongs to the unlabeled class, we make the pixel black..