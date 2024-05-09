import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from tabulate import tabulate

def plot_confusion_matrix(ax, conf_matrix, title, cmap):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title)

def draw_confusion_matrix(matrix,name):
  colormap=["Blues","Greens","Oranges"]
  fig, axes = plt.subplots(1, 1, figsize=(15, 5))
  for i in range(len(matrix)):
    plot_confusion_matrix(axes, matrix[i], "Confusion Matrix  ", colormap[i])
  plt.tight_layout()
  
#### function for plotting tables
def print_table(data,headers):
  table_data = list(zip(*data))
  table = tabulate(table_data, headers=headers, tablefmt='grid')
  print(table)

#function to calculate the size of machine learning models    
def evaluate_NN_models(model_list,X_test,y_test):
  loss, accuracy , y_pred ,precision, recall ,f1_score ,support ,confusion_matrix_list = [], [], [], [], [], [], [], []
  for model in model_list:
    loss_value,accuracy_value=model.evaluate(X_test, y_test)
    loss.append(loss_value)
    accuracy.append(accuracy_value)
    y_pred_value=np.argmax(model.predict(X_test), axis = 1)
    y_pred.append(y_pred_value)
    y_test_classes = y_test
    print("########",y_test_classes)
    print("$$$$$$$$",y_pred_value)
    precision_value, recall_value, f1_score_value, support_value = precision_recall_fscore_support(y_test_classes , y_pred_value, average= 'weighted' )
    precision.append(precision_value)
    recall.append(recall_value)
    f1_score.append(f1_score_value)
    support.append(support_value)
    confusion_matrix_value=confusion_matrix(y_test_classes, y_pred_value)
    confusion_matrix_list.append(confusion_matrix_value)
  return [accuracy ,precision, recall ,f1_score],confusion_matrix_list,y_pred