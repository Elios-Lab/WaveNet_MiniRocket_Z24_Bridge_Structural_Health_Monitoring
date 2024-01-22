from datasetManagement import datasetManagement #library DatasetManagement.py
from WaveNet import WavenetRun #library WaveNet.py
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Print information about each GPU
    for gpu in gpus:
        print("GPU:", gpu.name)
else:
    print("No GPU devices found.")

#Insert here the classes you want to use for the training, the first element will be labeled as 0, the second as 1 and so on
#values from 01 to 17
#classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']
classes = ['01', '03', '04', '05', '06']

X_train, X_val, X_test, y_train, y_val, y_test, width= datasetManagement(classes,65536)
model = None
#to load a model:
#model = tf.keras.models.load_model(path_model)
learning_rate=0.0001
filter=8
batchsize=2
epochs=2000
numberOfResidualsPerBlock=9 #2^(0), 2^(1),...,2^(N-1) 
numberOfBlocks=1 #E.g. equal to 2 : 2^0,..2^9,2^0,..2^9

WavenetRun(model,filter,batchsize,epochs,learning_rate,numberOfResidualsPerBlock,numberOfBlocks,width,classes,X_train, X_val, X_test, y_train, y_val, y_test)

    