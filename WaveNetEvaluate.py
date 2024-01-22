from datasetManagement import datasetManagement #library DatasetManagement.py
import utils #library utils.py
import tensorflow as tf

#Insert here the classes you want to use for the training, the first element will be labeled as 0, the second as 1 and so on
#values from 01 to 17
#classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']
classes = ['01', '03', '04', '05', '06']

X_train, X_val, X_test, y_train, y_val, y_test, width= datasetManagement(classes,65536)
model_path=""#model path here
model = tf.keras.models.load_model(model_path)
    
result_all_models_FC,confusion_matrix_FC,predicitions_FC=utils.evaluate_NN_models([model],X_test,y_test)
headers=['models','accuracy','precision','recall','f1_score']
data_normal_FC=[["WaveNet"]]
data_normal_FC.extend(result_all_models_FC)
utils.print_table(data_normal_FC,headers) 
utils.draw_confusion_matrix(confusion_matrix_FC,"WaveNet model")