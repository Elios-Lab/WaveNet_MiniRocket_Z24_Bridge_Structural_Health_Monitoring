import os
import numpy as np
import scipy.io
import random
from sktime.classification.kernel_based import RocketClassifier
import time
from tabulate import tabulate
import sklearn
from sklearn.model_selection import train_test_split
# Set seed to have the same dataset split and for reproducibility
seedNumber = 4
np.random.seed(seedNumber)
random.seed(seedNumber)

  
#### function for plotting tables
def print_table(data,headers):
  table_data = list(zip(*data))
  table = tabulate(table_data, headers=headers, tablefmt='grid')
  print(table)

  
def evaluate_MiniRocket_model(model,X_test,y_test):
    accuracy , y_pred ,precision, recall ,f1_score ,support ,confusion_matrix_list = [], [], [], [], [], [], []
    print("MiniRocket Model Predicting...")
    y_pred = model.predict(X_test)
    print("MiniRocket Model Predicting Finished.")
    
    accuracy_value = sklearn.metrics.accuracy_score(y_test, y_pred)

    accuracy.append(accuracy_value)
    precision_value, recall_value, f1_score_value, support_value = sklearn.metrics.precision_recall_fscore_support(y_test , y_pred, average= 'weighted' )
    precision.append(precision_value)
    recall.append(recall_value)
    f1_score.append(f1_score_value)
    support.append(support_value)

    return [accuracy ,precision, recall ,f1_score]

def datasetManagement(NbOfClasses,windows_length=65536):
    if NbOfClasses is None:
        raise ValueError("Classes cannot be None.")        
    if NbOfClasses == 15:
        classes = ['01','03','04','05','06','07','09','10','11','12','13','14','15','16','17']
    else:
        classes = ['01', '03', '04', '05', '06']

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
    y_train = np.array([np.array(x) for x in y_train])
    y_test = np.array([np.array(x) for x in y_test])
    y_val = np.array([np.array(x) for x in y_val])
   
    return X_train, X_val, X_test, y_train, y_val, y_test, width

windowSize = 65536
NbOfClasses = 5

# MiniRocket Model Parameters
num_kernels = 9996
maxDilationsPerKernel = 64

X_train, X_val, X_test, y_train, y_val, y_test, width = datasetManagement(NbOfClasses,windowSize)


start_time = time.time()
model = RocketClassifier(num_kernels, max_dilations_per_kernel=maxDilationsPerKernel, rocket_transform='minirocket', use_multivariate='no', n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
end_time = time.time()

print("Validation set results:")
result_all_models_FC=evaluate_MiniRocket_model(model,X_val,y_val)
headers=['accuracy','precision','recall','f1_score']
data_normal_FC=[["model"]]
data_normal_FC.extend(result_all_models_FC)
print_table(data_normal_FC,headers)

print("Execution time: ", end_time - start_time)
