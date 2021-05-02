# SemanticSegmentationCityscapes
Deep Learning Large Project

## Quick Start
For the purpose of demonstration, we have provided several trained models and selected samples at this [link for download](). 

The models should be found under `./models_display` and the selected samples are under `./data_display`. 

```
streamlit run dashboard.py
```

Do make sure that streamlit is downloaded (`pip install streamlit` should typically be sufficient).

## Dataset for Training
Note that the dataset file structure should be as follows (for illustration purposes, we only included three cities). 
```
./data
├── gtFine
│   ├── lindau
│   ├── munster
│   └── ulm
├── leftImg8bit
│   ├── lindau
│   ├── munster
│   └── ulm
└── leftImg8bit_sequence
    ├── lindau
    ├── munster
    └── ulm
```
This is important for preversing the training and evaluation process of the model. 

Augmented images have also been included in the respective directories and they are denoted by the sequence number (the first number in the filename) being higher than 900,000. 
