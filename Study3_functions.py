import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


elecs = ['C3','C4','P3','P4','F3','F4'] #'O1','O2' for control
central_elecs = elecs[0:2]
parietal_elecs = elecs[2:4]
frontal_elecs = elecs[4:6]
pair_elecs = [central_elecs, parietal_elecs, frontal_elecs]
name_pairs = ['central area','parietal area','frontal area']
name_ipsi_contra = ['central ipsilateral', 'parietal ipsilateral', 'frontal ipsilateral', 'central contralateral', 'parietal contralateral', 'frontal contralateral']
leftElects = ['C3','P3','F3']
rightElects = ['C4','P4','F4']
lateralElects = ['C ipsilateral','C contralateral','P ipsilateral','P contralateral', 'F ipsilateral', 'F contralateral']
lateral = ['I','C']
conditions = ['A','B','C']
subjects = ['001','010','011','012','013','002','044','005','006','007','008','009']
attend = ['A','U']
side = ['L','R']

TW_50ms = [[0,50],[50,100],[100,150],[150,200],[200,250],[250,300],[300,350],[350,400],[400,450],[450,500],[500,550],[550,600],[600,650],[650,700],[700,750],[750,800]]
TW_100ms = [[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,800]]
TW_200ms = [[0,200],[200,400],[400,600],[600,800]]
time_windows = [TW_50ms,TW_100ms,TW_200ms]

time_steps = 800 # in ms
bins = 800/2 # 1 bin 2 ms
bin_size = 5
number_bins = bins/bin_size
TW_lengths = [50,100,200]

classifiers = ['Random Forest', 'Bernoulli NB', 'Gaussian NB', 'Logistic Regression']

def selection (df,column,param):
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    for i in range(df.shape[0]):
        if param in df.iloc[i,df.columns.get_loc(column)]:
            df1=df1.append(df.iloc[i])
    return df1

def create_features_labels(df_bins, times):
    start_time = times[0]
    end_time = times[1]
    df_features_labels = pd.DataFrame()
    for s in subjects:
        df_subject=selection(df_bins,'subject',s)
        for c in conditions:
            labels = []
            if c=='A' : labels.append(1)
            if c=='B' : labels.append(2)
            if c=='C' : labels.append(3)
            df_condition=selection(df_subject,'condition',c)
            for a in attend:
                df_attend=selection(df_condition,'attend',a)
                for d in side:
                    df_side=selection(df_attend,'side',d)
                    numb_trials=df_side.shape[0]/len(elecs)
                    for i in range(numb_trials):
                        df1=pd.DataFrame()
                        df2=pd.DataFrame()
                        df1=(df_side.loc[df_side['trial'] == i]) #select data of the same trial number
                        df2['subject'] = pd.Series([s])
                        df2['condition'] = pd.Series([c])
                        df2['attend'] = pd.Series([a])
                        df2['side'] = pd.Series([d])
                        df2['trial'] = pd.Series([i])
                        df2['label'] = pd.Series([labels])
                        for e in elecs:
                            df_elecs=selection(df1,'electrode',e)
                            feature_elec = df_elecs.iloc[:,df_elecs.columns.get_loc("bin"+" "+str(start_time/10)):df_elecs.columns.get_loc("bin"+" "+str((end_time/10)-1))+1].values.tolist()
                            for k in range(len(feature_elec[0])):
                                df2[str(e)+' '+'feature'+' '+str(k)] = pd.Series([feature_elec[0][k]])
                        df_features_labels=df_features_labels.append(df2)
    return df_features_labels

def analysing():
    print('What do you want to analyse? \n 0: Classification of data per intervals \n 1: Feature importance per electrodes \n 2: Feature importance per lateral \n 3: Classification by laterality \n 4: Test ERPs and ERPs by laterality')
    selected=input()
    return selected

def tw_selection(windows_name):
    for i in range(len(windows_name)):
        print (str(i)+':'+' '+str(windows_name[i])+' '+'ms'+' '+'windows')
    print (" \n Note if the interval can be divided by the size of the selected time windows")
    selected=input()
    return selected

def select_time():
    print('Introduce start time in milliseconds: \n')
    start = input()
    print('Introduce end time in milliseconds: \n')
    end = input()
    times = [start, end]
    return times

def classify(df_feat_labels, TW_size, times):
    length_feature = (bins/bin_size)/(time_steps/TW_size)-1 
    df_classified = pd.DataFrame()
    df_classified['Classifiers'] = pd.Series(classifiers)
    for s in subjects:
        df_subject=selection(df_feat_labels,'subject',s)
        for n in range(time_steps/TW_size):
            features = []
            label = []
            if (n>=(times[0]/TW_size) and n<(times[1]/TW_size)):
                start = n*(number_bins/(time_steps/TW_size))
                end = n*(number_bins/(time_steps/TW_size))+length_feature
                for k in range(df_subject.shape[0]):
                    feature_TW = []
                    for e in elecs:
                        feature_TW_elec = (df_subject.iloc[k,df_subject.columns.get_loc(str(e)+' '+'feature'+' '+str(start)):df_subject.columns.get_loc(str(e)+' '+'feature'+' '+str(end))+1]).values.tolist()
                        feature_TW.extend(feature_TW_elec)
                    features.append(feature_TW)
                    label_TW= df_subject.iloc[k,df_subject.columns.get_loc('label')]
                    label.append(label_TW[0])
                classified_data = classified_tw(features,label)
                df_classified[str(s)+' '+str(start*10)+'-'+str((end+1)*10)+' '+'ms']=pd.Series(classified_data)
    return df_classified

def classified_tw (feature_TW,label_TW):
    feature_train, feature_test, label_train, label_test = train_test_split(feature_TW, label_TW, test_size=0.1, random_state=42)
    classif_TW = []
    counter = 1000
    accuracy_mean = []
    for i in range(counter):
        rf = RandomForestClassifier()
        accuracy_RF = classifiers_training(rf, feature_train, feature_test, label_train, label_test)
        accuracy_mean.append(accuracy_RF)
    accuracy_mean = np.asarray(accuracy_mean)
    accuracy_coeff_RF = np.mean(accuracy_mean)
    classif_TW.append(accuracy_coeff_RF)
    BernNB = BernoulliNB()
    accuracy_coeff_BNB = classifiers_training(BernNB , feature_train, feature_test, label_train, label_test)
    classif_TW.append(accuracy_coeff_BNB)
    GaussNB = GaussianNB()
    accuracy_coeff_GNB = classifiers_training(GaussNB , feature_train, feature_test, label_train, label_test)
    classif_TW.append(accuracy_coeff_GNB)
    LogRegres = LogisticRegression()
    accuracy_coeff_LR = classifiers_training(LogRegres , feature_train, feature_test, label_train, label_test)
    classif_TW.append(accuracy_coeff_LR)
    return classif_TW

def classifiers_training(classif, F_train, F_test, L_train, L_test):
    classif.fit(F_train, L_train)
    predictions = classif.predict(F_test)  
    accuracy_coeff = accuracy_score(L_test,predictions)
    return accuracy_coeff

def df_averages_classifier(df_classified, selected_TW, times, TW_size):
    df_mean = pd.DataFrame()
    df_mean['Classifiers'] = pd.Series(classifiers)
    df_total_mean = pd.DataFrame()
    df_total_mean[' '] = pd.Series('Total averages')
    analysis_start_time = times[0]
    analysis_end_time = times[1]
    TW_list = time_windows[selected_TW][(analysis_start_time/TW_size):(analysis_end_time/TW_size)]
    for t in TW_list:
        mean_per_TW = []
        total_mean_TW = []
        for i in range(len(classifiers)):
            TW_accuracies = []
            for s in subjects:
                acurracy_coeff_classif = (df_classified.iloc[i,df_classified.columns.get_loc(str(s)+' '+str(t[0])+'-'+str(t[1])+' '+'ms')]).tolist()
                TW_accuracies.append(acurracy_coeff_classif)
            mean_per_classif = np.mean(np.asarray(TW_accuracies))
            mean_per_TW.append(mean_per_classif)
        total_mean_TW=np.mean(np.asarray(mean_per_TW))
        df_mean[str(t[0])+'-'+str(t[1])+' '+'ms'] = pd.Series(mean_per_TW)
        df_total_mean[str(t[0])+'-'+str(t[1])+' '+'ms'] = pd.Series(total_mean_TW)
    return df_mean, df_total_mean

def plotting(df_classif_mean, df_total_mean, selected_TW, TW_size, times):
    start_time=times[0]
    end_time=times[1]
    total_average = np.asarray(df_total_mean.iloc[0,1:].tolist())
    plt.bar(np.arange(len(total_average)),total_average)
    plt.ylim(0.33,(np.amax(total_average)+0.01))
    plt.title('Total average accuracy')
    plt.xticks(np.arange(len(total_average)), time_windows[selected_TW][(start_time/TW_size):(end_time/TW_size)])
    plt.xlabel('Time windows (ms)')
    plt.ylabel('Accuracy coefficient')
    for k in range(len(classifiers)):
        plt.figure()
        classif_values= np.asarray(df_classif_mean.iloc[k,1:].tolist())
        plt.bar(np.arange(len(classif_values)),classif_values)
        plt.ylim(0.33,(np.amax(classif_values)+0.01))
        plt.title(str(classifiers[k])+' '+'classifier')
        plt.xticks(np.arange(len(classif_values)), time_windows[selected_TW][(start_time/TW_size):(end_time/TW_size)])
        plt.xlabel('Time windows (ms)')
        plt.ylabel('Accuracy coefficient')
        
def global_averages(df_mean_classifier, df_total_mean, selectedTW):
    plt.figure()
    averages = []
    averages_name = classifiers[:]
    averages_name.append('Total averages')
    for k in range(df_mean_classifier.shape[0]):
        averages.append(np.mean(np.asarray(df_mean_classifier.iloc[k,1:].tolist())))
    averages.append(np.mean(np.asarray(df_total_mean.iloc[0,1:].tolist())))
    
    x = np.arange(len(averages))
    plt.bar(x, averages, color='b', width=0.2)
    plt.ylim(0.33,0.43)
    plt.title('Classifiers')
    plt.xticks(np.arange(len(averages)), averages_name)
    plt.ylabel('Accuracy coefficient')

def analyse_feature_importance():
    print('Analyse: \n 0: Analyse one subject \n 1: Analyse all subjects')
    selected=input()
    return selected

def feature_importance(df_features_labels, times):
    bins_start = times[0]/10
    bins_end = times[1]/10
    df_importances = pd.DataFrame()
    all_subject_importances = []
    for s in subjects:
        features = []
        labels = []
        df_importances = pd.DataFrame()
        df_feat_label_subj = selection(df_features_labels,'subject',s)
        for k in range(df_feat_label_subj.shape[0]):
            feat_elecs = []
            for e in elecs:
                feat_elecs.extend((df_feat_label_subj.iloc[k,df_feat_label_subj.columns.get_loc(str(e)+' '+'feature'+' '+str(bins_start)):df_feat_label_subj.columns.get_loc(str(e)+' '+'feature'+' '+str(bins_end-1))+1]).values.tolist())
            features.append(feat_elecs)
            labels.append(df_feat_label_subj.iloc[k,df_feat_label_subj.columns.get_loc('label')][0])
        feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.1, random_state=42)
        counter = 1000
        importances = []
        for i in range(counter):
            rf = RandomForestClassifier()
            rf.fit(feature_train, label_train)
            importances.append(rf.feature_importances_) 
        average_importance = []
        for m in range(len(importances[0])):
            feat_import = []
            for j in range(counter):
                feat_import.append(importances[j][m])
            average_importance.append(np.mean(np.asarray(feat_import)))
        bins = len(average_importance)/len(elecs) 
        for e in range(len(elecs)):
            df_importances[str(elecs[e])] = pd.Series(average_importance[(e*bins):((e+1)*bins)])
        all_subject_importances.append(df_importances)
    return all_subject_importances

def selection_subject(subjects):
    for i in range(len(subjects)):
        print (str(i)+':'+' '+str(subjects[i])+' '+'subject')
    selected=input()
    return selected

def filterering_plotting_feature(df_subj_importance, times):
    bins_number = (times[1]-times[0])/10
    importance_values=[]
    for i in range(len(elecs)):
        importance_values.append(df_subj_importance.iloc[:,df_subj_importance.columns.get_loc(str(elecs[i]))].tolist())
    for k in range(len(importance_values)):
        sequence_activ = importance_values[k]
        bins = range(len(sequence_activ))
        plt.plot(bins,sequence_activ)
    plt.legend(elecs, loc='upper left')
    plt.show()
    plt.figure()
    k = 8 # change this value
    plt.title('Smoothing by averaging. K value '+str(k))
    for iv in range(len(elecs)): 
        filtered_elec = []
        elec_importance = importance_values[iv]
        for e in range(len(elec_importance)):
            if e<k:
                filtered_elec.append(sum(elec_importance[0:(e+k)+1])/(len(elec_importance[0:(e+k)+1])))
            elif e>(bins_number-k):
                filtered_elec.append(sum(elec_importance[(e-k):bins_number])/(len(elec_importance[(e-k):bins_number])))
            else:
                filtered_elec.append(sum(elec_importance[(e-k):(e+k)+1])/(len(elec_importance[(e-k):(e+k)+1])))
        plt.plot(bins,filtered_elec)
    plt.legend(elecs, loc='upper left')
    plt.show()
    
def filterering_plotting_feature_allsubj(all_df_feat, times):
    bins_number=(times[1]-times[0])/10
    df_features_subjects = pd.DataFrame()
    for e in elecs:
        all_features_elecs = np.zeros(bins_number)
        for i in range(len(subjects)):
            subj_df=all_df_feat[i]
            elec_features = np.asarray(subj_df[str(e)].values)
            all_features_elecs += elec_features
        df_features_subjects[str(e)]=pd.Series(all_features_elecs)
    filterering_plotting_feature(df_features_subjects, times)
    
def create_features_labels_lateral(df_bins, times):
    start_time = times[0]
    end_time = times[1]
    features_labels_all_lateral = []
    for l in lateral: #first ipsi, then contra
        features_labels_lateral = []
        df_leftElects = pd.DataFrame()
        df_rightElects = pd.DataFrame()
        df_lateral=selection(df_bins,'lateral',l)
        for s in subjects:
            df_subject=selection(df_lateral,'subject',s)
            for c in conditions:
                labels = []
                if c=='A' : labels.append(1)
                if c=='B' : labels.append(2)
                if c=='C' : labels.append(3)
                df_condition=selection(df_subject,'condition',c)
                for a in attend:
                    df_attend=selection(df_condition,'attend',a)
                    for d in side:
                        df_side=selection(df_attend,'side',d)
                        if ((d == 'L') and (l == 'I')) or ((d == 'R') and (l == 'C')):
                            numb_trials=df_side.shape[0]/len(leftElects)
                            for i in range(numb_trials):
                                df1=pd.DataFrame()
                                df2=pd.DataFrame()
                                df1=(df_side.loc[df_side['trial'] == i]) #select data of the same trial number
                                df2['subject'] = pd.Series([s])
                                df2['condition'] = pd.Series([c])
                                df2['attend'] = pd.Series([a])
                                df2['side'] = pd.Series([d])
                                df2['trial'] = pd.Series([i])
                                df2['label'] = pd.Series([labels])
                                for e in leftElects:
                                    df_elecs=selection(df1,'electrode',e)
                                    feature_elec = df_elecs.iloc[:,df_elecs.columns.get_loc("bin"+" "+str(start_time/10)):df_elecs.columns.get_loc("bin"+" "+str((end_time/10)-1))+1].values.tolist()
                                    for k in range(len(feature_elec[0])):
                                        df2[str(e)+' '+'feature'+' '+str(k)] = pd.Series([feature_elec[0][k]])
                                df_leftElects=df_leftElects.append(df2)    
                        else:
                            numb_trials=df_side.shape[0]/len(rightElects)
                            for i in range(numb_trials):
                                df1=pd.DataFrame()
                                df2=pd.DataFrame()
                                df1=(df_side.loc[df_side['trial'] == i])
                                df2['subject'] = pd.Series([s])
                                df2['condition'] = pd.Series([c])
                                df2['attend'] = pd.Series([a])
                                df2['side'] = pd.Series([d])
                                df2['trial'] = pd.Series([i])
                                df2['label'] = pd.Series([labels])
                                for e in rightElects:
                                    df_elecs=selection(df1,'electrode',e)
                                    feature_elec = df_elecs.iloc[:,df_elecs.columns.get_loc("bin"+" "+str(start_time/10)):df_elecs.columns.get_loc("bin"+" "+str((end_time/10)-1))+1].values.tolist()
                                    for k in range(len(feature_elec[0])):
                                        df2[str(e)+' '+'feature'+' '+str(k)] = pd.Series([feature_elec[0][k]])
                                df_rightElects=df_rightElects.append(df2)
        features_labels_lateral.append(df_leftElects)
        features_labels_lateral.append(df_rightElects)
        features_labels_all_lateral.append(features_labels_lateral)
    return features_labels_all_lateral

def feature_importance_laterality(features_labels_all_lateral, times):
    bins_nums = (times[1]-times[0])/10-1
    feature_imprt_all_lateral = []
    for i in range(len(features_labels_all_lateral)):
        lateral_all_elecs = features_labels_all_lateral[i]
        feature_imprt_all_elecs = []
        for j in range(len(lateral_all_elecs)):
            if j == 0:
                electrodes=leftElects
            else:
                electrodes=rightElects
            df_lateral_sideElects = lateral_all_elecs[j]
            features = []
            labels = []
            df_importances = pd.DataFrame()
            all_subject_importances = []
            for s in subjects:
                features = []
                labels = []
                df_importances = pd.DataFrame()
                df_lateral_sideElects_subj= selection(df_lateral_sideElects,'subject',s)
                for k in range(df_lateral_sideElects_subj.shape[0]):
                    feat_elecs = []
                    for e in electrodes:
                        feat_elecs.extend((df_lateral_sideElects_subj.iloc[k,df_lateral_sideElects_subj.columns.get_loc(str(e)+' '+'feature'+' '+'0'):df_lateral_sideElects_subj.columns.get_loc(str(e)+' '+'feature'+' '+str(bins_nums))+1]).values.tolist())
                    features.append(feat_elecs)
                    labels.append(df_lateral_sideElects_subj.iloc[k,df_lateral_sideElects_subj.columns.get_loc('label')][0])
                feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.1, random_state=42)
                counter = 1000
                importances = []
                for i in range(counter):
                    rf = RandomForestClassifier()
                    rf.fit(feature_train, label_train)
                    importances.append(rf.feature_importances_) 
                average_importance = []
                for m in range(len(importances[0])):
                    feat_import = []
                    for j in range(counter):
                        feat_import.append(importances[j][m])
                    average_importance.append(np.mean(np.asarray(feat_import)))
                bins = len(average_importance)/len(electrodes)
                for e in range(len(electrodes)):
                    df_importances[str(electrodes[e])] = pd.Series(average_importance[(e*bins):((e+1)*bins)])
                all_subject_importances.append(df_importances)
            feature_imprt_all_elecs.append(all_subject_importances)
        feature_imprt_all_lateral.append(feature_imprt_all_elecs)
    return feature_imprt_all_lateral

def feature_all_lateral_selection(feature_imprt_all_lateral, selected_subject):
    feature_all_lateral_selected = []
    for i in feature_imprt_all_lateral:
        feature_all_lateral_side = []
        for k in i:
            df_subject=k[selected_subject]
            feature_all_lateral_side.append(df_subject)
        feature_all_lateral_selected.append(feature_all_lateral_side)
    return feature_all_lateral_selected

def filterering_plotting_feature_laterality(feature_imprt_all_lateral_selec, times):
    bins_number = (times[1]-times[0])/10
    for l,k in enumerate(lateral):
        lateral_features=feature_imprt_all_lateral_selec[l]
        importance_values = []
        importance_values_left = []
        importance_values_right = []
        if k == 'I':
            df_subj_left = lateral_features[0]
            df_subj_right = lateral_features[1]
            for i in range(len(leftElects)):
                importance_values_left.append(np.asarray(df_subj_left.iloc[:,df_subj_left.columns.get_loc(str(leftElects[i]))].tolist()))
            for j in range(len(rightElects)):
                importance_values_right.append(np.asarray(df_subj_right.iloc[:,df_subj_right.columns.get_loc(str(rightElects[j]))].tolist()))
            importance_values_ipsi = []
            for h in range(len(elecs)/2):
                importance_values_ipsi.append(importance_values_left[h]+importance_values_left[h])
            for n in range(len(importance_values_ipsi)):
                sequence_activ = importance_values_ipsi[n]
                bins = range(len(sequence_activ))
                time = [i * 10 for i in bins]
                plt.plot(time,sequence_activ)
        elif k == 'C':
            df_subj_left = lateral_features[0]
            df_subj_right = lateral_features[1]
            for i in range(len(leftElects)):
                importance_values_left.append(np.asarray(df_subj_left.iloc[:,df_subj_left.columns.get_loc(str(leftElects[i]))].tolist()))
            for j in range(len(rightElects)):
                importance_values_right.append(np.asarray(df_subj_right.iloc[:,df_subj_right.columns.get_loc(str(rightElects[j]))].tolist()))
            importance_values_contra = []
            for h in range(len(elecs)/2):
                importance_values_contra.append(importance_values_left[h]+importance_values_left[h])
            for n in range(len(importance_values_contra)):
                sequence_activ = importance_values_contra[n]
                bins = range(len(sequence_activ))
                time = [i * 10 for i in bins]
                plt.plot(time,sequence_activ)   
        plt.legend(lateralElects, loc='upper left')
        plt.show()
    importance_values.append(importance_values_ipsi)
    importance_values.append(importance_values_contra)
    plt.figure()
    k = 8 # change this value
    plt.title('Smoothing by averaging. K value '+str(k))
    for imp_value in importance_values:
        for iv in range(len(imp_value)): #only central elecs
            filtered_elec = []
            elec_importance = imp_value[iv]
            for e in range(len(elec_importance)):
                if e<k:
                    filtered_elec.append(sum(elec_importance[0:(e+k)+1])/(len(elec_importance[0:(e+k)+1])))
                elif e>(bins_number-k):
                    filtered_elec.append(sum(elec_importance[(e-k):bins_number])/(len(elec_importance[(e-k):bins_number])))
                else:
                    filtered_elec.append(sum(elec_importance[(e-k):(e+k)+1])/(len(elec_importance[(e-k):(e+k)+1])))
            plt.plot(time,filtered_elec)
    plt.legend(lateralElects, loc='upper left')
    plt.show()
    
def filterering_plotting_feature_allsubj_laterality(feature_imprt_all_lateral, times):
    bins_num=(times[1]-times[0])/10
    importance_all_lateral = []
    for f,ind_lat in enumerate(feature_imprt_all_lateral): # i=0 ipsilateral & i=1 contralateral
        importance_all_lateral_side = []
        for k, ind_side in enumerate(ind_lat): # k=0 left & k=1 right
            df_all_subjects = pd.DataFrame()
            if k==0:
                electrodes = leftElects
            else:
                electrodes = rightElects
            for e in electrodes:
                all_features_elecs = np.zeros(bins_num)
                for i in range(len(subjects)):
                    subj_df=ind_side[i]
                    elec_features = np.asarray(subj_df[str(e)].values)
                    all_features_elecs += elec_features
                df_all_subjects[str(e)]=pd.Series(all_features_elecs)
            importance_all_lateral_side.append(df_all_subjects)
        importance_all_lateral.append(importance_all_lateral_side)
    filterering_plotting_feature_laterality(importance_all_lateral, times)
    
def separate_fl_laterality(df_feature_label):
    right_features = []
    left_features = []
    for k in leftElects:
        for i in range(number_bins):
            left_features.append(str(k)+' '+'feature'+' '+str(i))
    for k in rightElects:
        for i in range(number_bins):
            right_features.append(str(k)+' '+'feature'+' '+str(i))
    df_fl_left=selection(df_feature_label,'side','L')
    df_fl_left_ipsi= df_fl_left.drop(right_features, axis=1)
    df_fl_left_contra= df_fl_left.drop(left_features, axis=1)
    df_fl_right=selection(df_feature_label,'side','R')
    df_fl_right_ipsi= df_fl_right.drop(left_features, axis=1)
    df_fl_right_contra= df_fl_right.drop(right_features, axis=1)
    ipsi_fl = [df_fl_left_ipsi, df_fl_right_ipsi]
    contra_fl = [df_fl_left_contra, df_fl_right_contra]
    return ipsi_fl, contra_fl

def classify_laterality(feat_lab_lateral, TW_size, times):
    length_feature = (bins/bin_size)/(time_steps/TW_size)-1 
    df_classified_ipsi = pd.DataFrame()
    df_classified_contra = pd.DataFrame()
    df_classified_ipsi['Classifiers'] = pd.Series(classifiers)
    df_classified_contra['Classifiers'] = pd.Series(classifiers)
    for count_lat,i in enumerate(feat_lab_lateral):
        for s in subjects:
            df_subject_left=selection(i[0],'subject',s)
            df_subject_right=selection(i[1],'subject',s)
            for n in range(time_steps/TW_size):
                features = []
                labels = []
                if (n>=(times[0]/TW_size) and n<(times[1]/TW_size)):
                    start = n*(number_bins/(time_steps/TW_size))
                    end = n*(number_bins/(time_steps/TW_size))+length_feature
                    for j in range(df_subject_left.shape[0]):
                        feature_TW = []
                        label_TW = []
                        if count_lat == 0:
                            electrodes=leftElects
                        else:
                            electrodes=rightElects
                        for e in electrodes:
                            feature_TW_elec = (df_subject_left.iloc[j,df_subject_left.columns.get_loc(str(e)+' '+'feature'+' '+str(start)):df_subject_left.columns.get_loc(str(e)+' '+'feature'+' '+str(end))+1]).values.tolist()
                            feature_TW.extend(feature_TW_elec)
                        features.append(feature_TW)
                        label_TW= df_subject_left.iloc[j,df_subject_left.columns.get_loc('label')]
                        labels.append(label_TW[0])
                    for j in range(df_subject_right.shape[0]):
                        feature_TW = []
                        label_TW=[]
                        if count_lat == 0:
                            electrodes=rightElects
                        else:
                            electrodes=leftElects
                        for e in electrodes:
                            feature_TW_elec = (df_subject_right.iloc[j,df_subject_right.columns.get_loc(str(e)+' '+'feature'+' '+str(start)):df_subject_right.columns.get_loc(str(e)+' '+'feature'+' '+str(end))+1]).values.tolist()
                            feature_TW.extend(feature_TW_elec)
                        features.append(feature_TW)
                        label_TW= df_subject_right.iloc[j,df_subject_right.columns.get_loc('label')]
                        labels.append(label_TW[0])
                    classified_data = classified_tw(features,labels)
                    if count_lat ==0:
                        df_classified_ipsi[str(s)+' '+str(start*10)+'-'+str((end+1)*10)+' '+'ms']=pd.Series(classified_data)
                    elif count_lat ==1:
                        df_classified_contra[str(s)+' '+str(start*10)+'-'+str((end+1)*10)+' '+'ms']=pd.Series(classified_data)
    df_classified_lateral = [df_classified_ipsi,df_classified_contra]                 
    return df_classified_lateral

def lateral_selection():
    print ('Selection laterality: \n 0: Ipsilateral \n 1: Contralateral \n 2: Comparison between ipsilateral and contralateral')
    selected = input()
    return selected

def plotting_lateral(df_mean_classifier_ipsi, df_total_mean_ipsi, df_mean_classifier_contra, df_total_mean_contra, selectedTW, TW_size, times):
    total_average_ipsi = np.asarray(df_total_mean_ipsi.iloc[0,1:].tolist())
    total_average_contra = np.asarray(df_total_mean_contra.iloc[0,1:].tolist())
    x = np.arange(len(total_average_ipsi))
    plt.bar(x-0.2, total_average_ipsi, color='r', width=0.2)
    plt.bar(x, total_average_contra, color='b', width=0.2)
    plt.ylim(0.33,0.43)
    plt.title('Total average accuracy ipsilateral vs. contralateral')
    plt.xticks(np.arange(len(total_average_ipsi)), time_windows[selectedTW][(times[0]/TW_size):(times[1]/TW_size)])
    plt.xlabel('Time windows (ms)')
    plt.ylabel('Accuracy coefficient')
    for k in range(len(classifiers)):
        plt.figure()
        classif_values_ipsi= np.asarray(df_mean_classifier_ipsi.iloc[k,1:].tolist())
        classif_values_contra= np.asarray(df_mean_classifier_contra.iloc[k,1:].tolist())
        x = np.arange(len(classif_values_ipsi))
        plt.bar(x-0.2, classif_values_ipsi, color='r', width=0.2)
        plt.bar(x, classif_values_contra, color='b', width=0.2)
        plt.ylim(0.33,0.43)
        plt.title(str(classifiers[k])+' classifier: ipsilateral vs. contralateral')
        plt.xticks(np.arange(len(classif_values_ipsi)), time_windows[selectedTW][(times[0]/TW_size):(times[1]/TW_size)])
        plt.xlabel('Time windows (ms)')
        plt.ylabel('Accuracy coefficient')
        
def global_averages_lateral(df_mean_classifier_ipsi, df_total_mean_ipsi, df_mean_classifier_contra, df_total_mean_contra, selectedTW):
    averages_ipsi = []
    averages_contra = []
    averages_name = classifiers[:]
    averages_name.append('Total averages')
    for k in range(df_mean_classifier_ipsi.shape[0]):
        averages_ipsi.append(np.mean(np.asarray(df_mean_classifier_ipsi.iloc[k,1:].tolist())))
    for k in range(df_mean_classifier_contra.shape[0]):
        averages_contra.append(np.mean(np.asarray(df_mean_classifier_contra.iloc[k,1:].tolist())))
    averages_ipsi.append(np.mean(np.asarray(df_total_mean_ipsi.iloc[0,1:].tolist())))
    averages_contra.append(np.mean(np.asarray(df_total_mean_contra.iloc[0,1:].tolist())))
    x = np.arange(len(averages_ipsi))
    plt.bar(x-0.2, averages_ipsi, color='r', width=0.2)
    plt.bar(x, averages_contra, color='b', width=0.2)
    plt.ylim(0.33,0.43)
    plt.title('Classifiers ipsilateral vs. contralateral')
    plt.xticks(np.arange(len(averages_ipsi)), averages_name)
    plt.ylabel('Accuracy coefficient')
    
def testing_ERPs(all_bins):
    tasks = []
    for c in conditions:
        condit = []
        ipsilateral = []
        contralateral = []
        lateral_index = []
        df_condition = selection (all_bins, 'condition', c)    
        for l in lateral:
            df_lateral = selection (df_condition, 'lateral', l)
            for p in pair_elecs:
                ERPs = []
                df1=df_lateral.drop(df_lateral.index[[0,df_lateral.shape[0]-1]])
                for i in range(df_lateral.shape[0]):
                    if (p[0] or p[1]) in df_lateral.iloc[i,df_lateral.columns.get_loc('electrode')]:
                        df1=df1.append(df_lateral.iloc[i])
                for n in range(number_bins):
                    activations_per_trial = np.asarray(df1['bin'+' '+str(n)])
                    mean_activations = np.mean(activations_per_trial)
                    ERPs.append(mean_activations)
                if l == 'I':
                    ipsilateral.append(ERPs)
                elif l == 'C':
                    contralateral.append (ERPs)
        condit.append(ipsilateral)  
        condit.append(contralateral)          
        for j in range(len(ipsilateral)):
            ipsi_array = np.asarray(ipsilateral[j])
            contra_array = np.asarray(contralateral[j])
            LI_array = (ipsi_array-contra_array)/(ipsi_array+contra_array)
            lateral_index.append(LI_array)
        condit.append(lateral_index)
        tasks.append(conditions)
    position = 1
    for count_c, c in enumerate(conditions):
        cond = tasks[count_c]
        ipsilateral = cond [0]
        contralateral = cond [1]
        plt.subplot(1,3,position)
        for k in range(len(ipsilateral)):
            sequence_activ = ipsilateral [k]
            time = [i*10 for i in range(number_bins)]
            plt.plot(time, sequence_activ)
        for k in range(len(contralateral)):
            sequence_activ = contralateral [k]
            time = [i*10 for i in range(number_bins)]
            plt.plot(time, sequence_activ)
        plt.legend(name_ipsi_contra, loc='upper left')
        plt.ylim(-1,22)
        plt.title(str (c)+' task')
        position = position + 1
    plt.show()
    plt.figure()
    position = 1
    for count_c, c in enumerate(conditions):
        cond = tasks[count_c]
        lateral_index = cond [2]
        plt.subplot(1,3,position)
        for k in range(len(lateral_index)):
            sequence_activ = lateral_index [k]
            time = [i*10 for i in range(number_bins)]
            plt.plot(time, sequence_activ)
        plt.legend(name_pairs, loc='upper left')
        plt.ylim(-1,1)
        plt.title(str (c)+' task') 
        position = position + 1  
        