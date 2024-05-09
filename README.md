WaveNet MiniRocket Z24 Bridge Structural Health Monitoring
=================
### Table of contents
- [Overview](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#overview)
- [About the Z24 Bridge Dataset](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#about-the-Z24-Bridge-Dataset)
- [Installation](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#installation)    
    - [For Windows](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#for-windows)
    - [For Linux](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#for-linux)
- [Usage](https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git#Usage)
    
## Overview
This repository contains the implementation of a cutting-edge approach to Structural Health Monitoring (SHM) of the Z24 Bridge dataset, leveraging the powerful capabilities of WaveNet and MiniRocket algorithms. Our project aims to provide a comprehensive toolset for the analysis, prediction, and understanding of the structural integrity of the Z24 Bridge, a well-known case study in the field of civil engineering and Structural Health Monitoring.
## About the Z24 Bridge Dataset
The Z24 Bridge dataset is a crucial resource in the SHM community, offering extensive sensor readings from the now-demolished Z24 Bridge in Switzerland. This dataset includes data on various parameters like temperature, strain, and displacement, providing a rich source for analyzing structural behaviors and anomalies. For this work, the Progressive Damage Test (PDT) section is used, in particular the ambient vibration test (avt).

The dataset is not provided by this github and must be requested from the owner at the link: https://bwk.kuleuven.be/bwm/z24
## Installation

1. Clone this git repository in an appropriate folder
```
git clone https://github.com/Elios-Lab/WaveNet_MiniRocket_Z24_Bridge_Structural_Health_Monitoring.git
```

2. Request the dataset from the owner at the link: https://bwk.kuleuven.be/bwm/z24

3. Setup conda environment
```
conda env create -f Z24_Bridge_env.yml
conda activate BridgeZ24_1
```

4. Install the required packages
```
pip install tensorflow
pip install sktime
```
We suggest to install TensorFlow with GPU support if you have a compatible GPU.

5. Add the Dataset

Find in the downloaded dataset the files for Progressive Damage Test (PDT) and inside the folder Ambient Vibration Test (avt) and place each element in the correct folder: e.g. in DatasetPDT/01/avt/ copy files from `01setup01.mat` to `01setup09.mat`.
Do this for all classes from 1 to 17. 

For class 03 remove the file `03setup01.mat` as declared in the journal paper.

## Usage
Inside this repository you can find files relative to WaveNet and to the Minirocket deep neural networks.

The script are ready to be launched after the dataset has been correctly inserted in the folder.

The WaveNet is implemented in the file `WaveNet.py` with the construction of the model. 

The training can be started by running the `WavenetRun.py` script and the model can be evaluated through the `WaveNetEvaluate.py` script after specifying the model path in the script.

The Wavenet files `datasetManagement.py` and `utils.py` regard the preparation of the dataset and utility functions.
For the MiniRocket there is one single file `MiniRocket.py` which also implements a Ridge Classifier after MiniRocket's feature extraction.
In the folder `History` it will be saved the summary of the training at the end of a training session.

To select 5 or 15 classes (as declared in the journal paper) choose in the code one of these two lines:
```
#5 classes
classes = ['01', '03', '04', '05', '06']
#15 classes
classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']
```
