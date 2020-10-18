from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, average_precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
import torchnet.meter.confusionmeter as cm
from scipy import interp
from itertools import cycle
from PIL import Image
from collections import OrderedDict
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

matplotlib.use('Agg')

def calculateROC(target, pred, idx, class_list, epoch, save_dir, save=True, prefix=None):
    fpr, tpr, _ = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)

    img_name = 'epoch_{}_roc_{}.png'.format(str(epoch).zfill(3), class_list[idx])
    if isinstance(prefix, str):
        img_name = prefix + img_name

    if save:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(class_list[idx]))
        plt.legend(loc="lower right")
        plt.savefig(osp.join(save_dir,img_name), dpi=300)
        plt.close()
        #print('Epoch : {}, ROC curve on {} : {:.2f}'.format(epoch, class_list[idx], roc_auc))

    return roc_auc

def plotingConfusionMatrix(target, pred, class_list, epoch, save_dir, normalize=True, title=None, cmap=plt.cm.Blues, prefix=None):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    img_name = 'epoch_{}_confusionMatrix.png'.format(str(epoch).zfill(3))
    if isinstance(prefix, str):
        img_name = prefix + img_name

    # Compute confusion matrix
    cm = confusion_matrix(target, pred)
    cm1 = cm
    class_list = np.array(class_list)
    class_list = class_list[unique_labels(list(target), list(np.squeeze(pred)))]

    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100, 0)
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_list, yticklabels=class_list,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = int(cm.max() * 0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            num = format(cm[i, j], fmt)
            num = num.split(".")[0]
            ax.text(j, i, str(cm1[i, j]) + " (" + num + "%)",
                    ha="center", va="center",
                    #color="black", fontsize=8)
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(osp.join(save_dir, img_name), dpi=300)
    plt.close()

    return ax