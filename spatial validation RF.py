import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler


##################### read data #####################################################
# data_rock = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide rock spearman.csv")
# data_soil = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide soil spearman.csv")
data_rock = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide rock pearson.csv")
data_soil = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/landslide soil pearson.csv")
all_data = np.concatenate((data_rock, data_soil), axis=0)
#### scale the data to mean of 0, and std of 1
minmax = StandardScaler()
all_data_scale_x = minmax.fit_transform(all_data[:,:-1])
all_data_scale = np.concatenate((all_data_scale_x, all_data[:,-1].reshape(len(all_data), 1)), axis=1)
data_rock_scale = all_data_scale[:len(data_rock), :]
data_soil_scale = all_data_scale[len(data_rock):,:]
##########################################################################
test_rock_id = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/rock spatial test ID.csv").values.ravel()
test_soil_id = pd.read_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Codes/soil spatial test ID.csv").values.ravel()

############ selecte train and test samples based on spatial division
selected_train_rock_rows = np.ones(data_rock_scale.shape[0], dtype=bool)
selected_train_rock_rows[test_rock_id] = False
train_rock = data_rock_scale[selected_train_rock_rows,:]
test_rock = data_rock_scale[test_rock_id,:]


selected_train_soil_rows = np.ones(data_soil_scale.shape[0], dtype=bool)
selected_train_soil_rows[test_soil_id] = False
train_soil = data_soil_scale[selected_train_soil_rows,:]
test_soil = data_soil_scale[test_soil_id,:]

all_auc, all_acc, all_recall, all_precision, all_f1, all_kappa = [], [], [], [], [], []
for seed in range(1, 101):
    ## get a balanced train dataset
    train_rock_new, left_rock = train_test_split(train_rock, train_size=len(train_soil), shuffle=True,
                                             random_state=seed)
    train_data = np.concatenate((train_rock_new, train_soil), axis=0)
    all_train_x, all_train_y = train_data[:,:-1], train_data[:,-1]


    ## get the test dataset
    test_data = np.concatenate((test_rock, test_soil), axis=0)
    all_test_x, all_test_y = test_data[:,:-1], test_data[:,-1]


    ## build the model
    model = RandomForestClassifier(random_state=1)
    # model = svm.SVC(probability=True)
    # model = LogisticRegression()
    model.fit(all_train_x, all_train_y)


    ## evaluate the performance
    y_pred_label = model.predict(all_test_x)
    y_pred_prob = model.predict_proba(all_test_x)
    y_probability_first = [x[1] for x in y_pred_prob]


    test_auc = metrics.roc_auc_score(all_test_y, y_probability_first)
    test_acc = metrics.accuracy_score(all_test_y, y_pred_label)
    test_recall = metrics.recall_score(all_test_y, y_pred_label)
    test_precision = metrics.precision_score(all_test_y, y_pred_label)
    test_f1 = metrics.f1_score(all_test_y, y_pred_label)
    test_kappa = metrics.cohen_kappa_score(all_test_y, y_pred_label)
    print(test_acc, test_auc, test_recall, test_precision, test_f1, test_kappa)
    print('------------------------------------------------------------------------')

    all_auc.append(np.round(test_auc, 3))
    all_acc.append(np.round(test_acc, 3))
    all_recall.append(np.round(test_recall, 3))
    all_precision.append(np.round(test_precision, 3))
    all_f1.append(np.round(test_f1, 3))
    all_kappa.append(np.round(test_kappa, 3))


print(round(np.mean(all_auc), 3), round(np.mean(all_acc), 3), round(np.mean(all_recall), 3),
      round(np.mean(all_precision), 3), round(np.mean(all_f1), 3))
print(round(np.std(all_auc), 3), round(np.std(all_acc), 3), round(np.std(all_recall), 3),
      round(np.std(all_precision), 3), round(np.std(all_f1), 3))


out_result = pd.DataFrame(all_auc)*100
# out_result.columns = data_rock.iloc[:,:-1].columns
out_result.to_csv("C:/Users/DELL/OneDrive - cug.edu.cn/data_basalt/Results/all_auc_spatialV.csv",encoding="utf_8_sig", header=False, index=False)
# print(model.feature_importances_)

