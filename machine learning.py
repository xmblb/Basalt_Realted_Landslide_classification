import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler


##################### read data #####################################################
# data_rock = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide rock spearman.csv")
# data_soil = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide soil spearman.csv")
data_rock = pd.read_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/data_basalt/Codes/landslide rock pearson.csv")
data_soil = pd.read_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/data_basalt/Codes/landslide soil pearson.csv")
all_data = np.concatenate((data_rock, data_soil), axis=0)
#### scale the data to mean of 0, and std of 1
minmax = StandardScaler()
all_data_scale_x = minmax.fit_transform(all_data[:,:-1])
all_data_scale = np.concatenate((all_data_scale_x, all_data[:,-1].reshape(len(all_data), 1)), axis=1)
data_rock_scale = all_data_scale[:len(data_rock), :]
data_soil_scale = all_data_scale[len(data_rock):,:]
##########################################################################

all_prob = []
auc_label, auc_prob = [], []
all_auc, all_acc, all_recall, all_precision, all_f1, all_kappa = [], [], [], [], [], []
importance_score = []
for seed in range(1, 101):
    ## get a balanced dataset, select the same number of soilslide
    train_rock, left_rock = train_test_split(data_rock_scale, train_size=len(data_soil_scale), shuffle=True, random_state=seed)
    train_soil = data_soil_scale
    train_data = np.concatenate((train_rock, train_soil), axis=0)
    all_train_x, all_train_y = train_data[:,:-1], train_data[:,-1]


    ### 5-fold corss-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    fold_number = 1
    temp_auc, temp_acc, temp_recall, temp_precision, temp_f1, temp_kappa = [], [], [], [], [], []
    temp_importa = []
    temp_prob = np.array([])
    all_tn, all_fp, all_fn, all_tp = [], [], [], []
    for train_idx, test_idx in kfold.split(all_train_x, all_train_y):

        ## define the model
        # model = RandomForestClassifier(random_state=1)
        # model = svm.SVC(probability=True)
        model = LogisticRegression()
        model = MLPClassifier()
        model.fit(all_train_x[train_idx], all_train_y[train_idx])


        ########################################################################
        ########### prediction in a balanced test dataset
        y_pred_label = model.predict(all_train_x[test_idx])
        y_pred_prob = model.predict_proba(all_train_x[test_idx])
        y_probability_first = [x[1] for x in y_pred_prob]

        auc_label.append(all_train_y[test_idx])
        auc_prob.append(y_probability_first)

        test_auc = metrics.roc_auc_score(all_train_y[test_idx], y_probability_first)
        test_acc = metrics.accuracy_score(all_train_y[test_idx], y_pred_label)
        test_recall = metrics.recall_score(all_train_y[test_idx], y_pred_label)
        test_precision = metrics.precision_score(all_train_y[test_idx], y_pred_label)
        test_f1 = metrics.f1_score(all_train_y[test_idx], y_pred_label)
        test_kappa = metrics.cohen_kappa_score(all_train_y[test_idx], y_pred_label)

        temp_prob = np.concatenate((temp_prob, y_probability_first))


        tn, fp, fn, tp = metrics.confusion_matrix(all_train_y[test_idx], y_pred_label).ravel()
        tpr = tp*100/(tp+fn)
        fnr = fn*100/(fn+tp)
        fpr = fp*100/(fp+tn)
        tnr = tn*100/(fp+tn)


        all_tn.append(tnr)
        all_fp.append(fpr)
        all_fn.append(fnr)
        all_tp.append(tpr)

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_number} ...')
        # print(f'Training for fold {fold_number} ...')
        print(test_acc, test_auc, test_recall, test_precision, test_f1, test_kappa)
        fold_number += 1
        # temp_auc.append(np.round(test_auc, 3))
        # temp_acc.append(np.round(test_acc, 3))
        # temp_recall.append(np.round(test_recall, 3))
        # temp_precision.append(np.round(test_precision, 3))
        # temp_f1.append(np.round(test_f1, 3))
        # temp_kappa.append(np.round(test_kappa, 3))
        all_auc.append(np.round(test_auc, 3))
        all_acc.append(np.round(test_acc, 3))
        all_recall.append(np.round(test_recall, 3))
        all_precision.append(np.round(test_precision, 3))
        all_f1.append(np.round(test_f1, 3))
        all_kappa.append(np.round(test_kappa, 3))



        # importance_score.append(np.round(model.feature_importances_, 3))
    # all_auc.append(temp_auc)
    # all_acc.append(temp_acc)
    # all_recall.append(temp_recall)
    # all_precision.append(temp_precision)
    # all_f1.append(temp_f1)
    # all_kappa.append(temp_kappa)
    all_prob.append(temp_prob)

print(round(np.mean(all_auc), 3), round(np.mean(all_acc), 3), round(np.mean(all_recall), 3), round(np.mean(all_precision), 3),round(np.mean(all_f1), 3))
print(round(np.std(all_auc), 3), round(np.std(all_acc), 3), round(np.std(all_recall), 3), round(np.std(all_precision), 3),round(np.std(all_f1), 3))

print(round(np.mean(all_tp), 1), round(np.mean(all_fn), 1), round(np.mean(all_fp), 1), round(np.mean(all_tn), 1))
print(round(np.std(all_tp), 1), round(np.std(all_fn), 1), round(np.std(all_fp), 1), round(np.std(all_tn), 1))

out_result = pd.DataFrame(all_f1)*100
# out_result.columns = data_rock.iloc[:,:-1].columns
out_result.to_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Results/all_f1.csv",encoding="utf_8_sig", header=False, index=False)
# print(model.feature_importances_)


out_result = pd.DataFrame(all_prob)
# out_result.columns = data_rock.iloc[:,:-1].columns
out_result.to_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/data_basalt/Results/all_prob.csv",encoding="utf_8_sig", header=False, index=False)