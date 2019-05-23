import numpy as np
import matplotlib.pyplot as plt

def sliding_window(sequence,
                   win_size = 20, step = 5):
    nb_chunks = ((len(sequence)-win_size)//step)+1
    for i in range(0,nb_chunks*step,step):
        yield sequence[i:i+win_size]

def major_filter(activity):
    unique, counts = np.unique(activity, return_counts=True)
    return unique[np.argmax(counts)]

def one_hot(a, num_classes = 4):
    a = a.astype('int32') - 1
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.jet):
    """
        https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
    """
    plt.figure(figsize(20,20),facecolor='white')
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title("%s %s" % (title, model.__class__.__name__))
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name, fontsize=12)
    plt.xlabel(df_confusion.columns.name, fontsize=12)

def create_conf_matrix(y_true, y_pred, n_classes = 4):
    """
        src : https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
    """
    m = np.zeros([n_classes,n_classes],dtype=np.int32)
    for pred, exp in zip(y_pred,y_true):
        m[pred][exp] += 1
    return m