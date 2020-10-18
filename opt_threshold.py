import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from collections import defaultdict


class OptThreshold(object):
    def __init__(self):

        return

    def find_optimal_cutoff_multi(self, target, predicted):
        print(list(set(target)))
        unique_class = list(set(target))
        target = label_binarize(target, list(set(target)))

        # create class dict for better visualization
        name_dict = defaultdict(str)
        for i in range(len(unique_class)):
            name_dict[i] = str(unique_class[i]) + ' : '
        print(name_dict)
        for i in range(len(unique_class)):
            print("-" * 40)
            print(name_dict[i])
            self.find_optimal_cutoff(target[:,i], predicted[:, i])
            print("-" * 40)
        return

    def find_optimal_cutoff(self, target, predicted, txt=None, thres=None):

        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i),
                       'tpr' : pd.Series(tpr), 'fpr' : pd.Series(fpr) })
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        if not isinstance(thres, float):
            y_score_opt = np.where(predicted >= list(roc_t['threshold'])[0], 1, 0)
        else:
            y_score_opt = np.where(predicted >= thres, 1, 0)

        if isinstance(txt, str):
            if isinstance(thres, float):
                txt = 'thres_{}_'.format(thres) + txt

            with open(txt, 'w') as file_point:
                if isinstance(thres, float):
                    print('custom threshold : ', thres, file=file_point)
                else:
                    print('optimal threshold : ', list(roc_t['threshold'])[0], file=file_point)
                print('accuracy : ', accuracy_score(target, y_score_opt), file=file_point)
                print('auc : ', auc(fpr, tpr), file=file_point)
                #print('conf mat : \n', confusion_matrix(target, y_score_opt), file=fp)
                #tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel() # sensitivty / specificity / ppv / npv
                tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel() # sensitivty / specificity / ppv / npv
                print('tn : ', tn, 'fp : ', fp, 'fn : ', fn, 'tp : ', tp, file=file_point)
                print('sensitivity : ', tp / (tp + fn), file=file_point)
                print('specificity : ', tn / (fp + tn), file=file_point)
                print('PPV : ', tp / (tp + fp), file=file_point)
                print('NPV : ', tn / (tn + fn), file=file_point)
        else:
            if isinstance(thres, float):
                print('custom threshold : ', thres)
            else:
                print('optimal threshold : ', list(roc_t['threshold'])[0])
            print('accuracy : ', accuracy_score(target, y_score_opt))
            print('auc : ', auc(fpr, tpr))
            print('conf mat : \n', confusion_matrix(target, y_score_opt))
            #tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel()  # sensitivty / specificity / ppv / npv
            tn, fp, fn, tp = confusion_matrix(target, y_score_opt).ravel()  # sensitivty / specificity / ppv / npv
            print('tn : ', tn, 'fp : ', fp, 'fn : ', fn, 'tp : ', tp)
            print('sensitivity : ', tp / (tp + fn))
            print('specificity : ', tn / (fp + tn))
            print('PPV : ', tp / (tp + fp))
            print('NPV : ', tn / (tn + fn))

        return accuracy_score(target, y_score_opt), y_score_opt