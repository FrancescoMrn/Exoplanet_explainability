'''
Inside utils.py are reported those functions that used to handle input, output and
results.
'''

import numpy as np

def data_loader_txt(path, label_column_index, skiprows=1, delimiter=',', normalize=True):
    '''
    Light-weight txt files data loader that uses numpy as backend
    '''
    raw = np.loadtxt(path, skiprows=1, delimiter=',')
    x = raw[:, label_column_index+1:]
    y = raw[:, label_column_index, np.newaxis] - 1. # -1 to report label in the range 0-1
    
    if normalize: # standard normalization
        x = ((x - np.mean(x, axis=1).reshape(-1,1)) / np.std(x, axis=1).reshape(-1,1))
    return x, y


def plot_confusion_matrix(cm,target_names, title='Confusion matrix', cmap=None, normalize=True):
    '''
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n \nAccuracy={:0.3f}; Missclass={:0.3f}'.format(accuracy, misclass))
    plt.show()