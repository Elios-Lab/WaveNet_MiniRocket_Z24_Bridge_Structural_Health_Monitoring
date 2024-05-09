import os
import numpy as np
import scipy.io
import tensorflow as tf
import random
#Seed to split the dataset in the same way for WaveNetRun and WaveNetEvaluate and for reproducibility
np.random.seed(4)
random.seed(4)
from sklearn.model_selection import train_test_split

def datasetManagement(classes,windows_length=65536):
    if classes is None:
        raise ValueError("Classes cannot be None.")        

    classesPaths=[]
    for classNum in classes:
        classesPaths.append('./DatasetPDT/'+classNum+'/avt/')
    print(classesPaths)
    lenTimeserie = 65536

    #Load data
    datasetFull = []
    labelsFull = []
    for classPath in classesPaths:
        print(classPath)
        for path in os.listdir(classPath):
            if path.endswith(".mat"):
                mat = scipy.io.loadmat(classPath+path)
                dataAggregated = mat['data']
                dataAggregated = dataAggregated.T

                for i in range(len(dataAggregated)):
                    #padding to lenTimeserie
                    if(len(dataAggregated[i])<lenTimeserie):
                        npData=np.array(dataAggregated[i])
                        last_value = npData[-1]
                        additional_values = np.full(lenTimeserie - len(npData), last_value)
                        npData = np.concatenate((npData, additional_values))                        
                        datasetFull.append(np.array(npData))
                    else:
                        #add directly to data
                        datasetFull.append(np.array(dataAggregated[i]))
                    #add label
                    labelsFull.append(classesPaths.index(classPath))


    # Combine dataset and labels for shuffling
    combined_data = list(zip(datasetFull, labelsFull))
    from sklearn.utils import shuffle
    # Shuffle the combined data
    shuffled_data = shuffle(combined_data, random_state=42)
    combined_data=None
    # Split the shuffled data back into dataset and labels
    shuffled_dataset, shuffled_labels = zip(*shuffled_data)
    shuffled_dataset = np.array(shuffled_dataset)
    shuffled_labels = np.array(shuffled_labels)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Scale and reshape dataset
    scaled_dataset = scaler.fit_transform(shuffled_dataset.reshape(-1, lenTimeserie)).reshape(shuffled_dataset.shape)

    combined_data = list(zip(scaled_dataset, shuffled_labels))
    train_data, test_data, val_data = [], [],[]

    # 70% train, 20% test, 10% validation
    for class_label in range(len(classes)):
        class_data, class_labels = zip(*[(data, label) for data, label in combined_data if label == class_label])
        train, test, ytrain, ytest = train_test_split(class_data, class_labels, test_size=1/5, random_state=42)
        train, val, ytrain, yval = train_test_split(train, ytrain, test_size=1/8, random_state=42)
        train_data.extend(zip(train, ytrain))
        test_data.extend(zip(test, ytest))
        val_data.extend(zip(val, yval))

    # Unzip and convert the data into numpy arrays
    X_train_pre, y_train_pre = map(np.array, zip(*train_data))
    X_test_pre, y_test_pre = map(np.array, zip(*test_data))
    X_val_pre, y_val_pre = map(np.array, zip(*val_data))

    X_train_pre = np.array(X_train_pre)
    X_test_pre = np.array(X_test_pre)
    X_val_pre = np.array(X_val_pre)
    y_train_pre = np.array(y_train_pre)
    y_test_pre = np.array(y_test_pre)
    y_val_pre = np.array(y_val_pre)

    # for windowing (with full length sequence will not change the data)
    def create_windows(data_array, label_array, windows_length): #Generate from 65K to N windows of selected length
        X, y = [], []
        for i in range(len(data_array)):
            for start in range(0, len(data_array[i]) - windows_length + 1, windows_length):
                end = start + windows_length
                X.append(data_array[i][start:end])
                y.append(label_array[i])
        return X, y
        
    X_train, y_train = create_windows(X_train_pre, y_train_pre, windows_length)
    X_val, y_val = create_windows(X_val_pre, y_val_pre, windows_length)
    X_test, y_test = create_windows(X_test_pre, y_test_pre, windows_length)
    
    width = len(X_train[0])
 
    X_train = np.array([np.array(x) for x in X_train])
    X_val = np.array([np.array(x) for x in X_val])
    X_test = np.array([np.array(x) for x in X_test])
    
    # Reshape the input data to have a third dimension
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)     
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(X_train.shape,X_val.shape,X_test.shape)   
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test, width