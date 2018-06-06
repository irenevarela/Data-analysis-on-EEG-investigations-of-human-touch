import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
import Study3_functions

path='C:\Users\Irene\Documents\EEG/'
all_bins =cPickle.load(open('AllBins0-800','rb'))

times = Study3_functions.select_time()
df_feature_label = Study3_functions.create_features_labels(all_bins, times)
analysing = Study3_functions.analysing()
TW_lengths = [50,100,200]
subjects = ['001','010','011','012','013','002','044','005','006','007','008','009']

if analysing == 0: # Classification of data per intervals (general) 
    selectedTW = Study3_functions.tw_selection(TW_lengths)
    df_classification = Study3_functions.classify(df_feature_label, TW_lengths[selectedTW], times)
    df_mean_classifier, df_total_mean = Study3_functions.df_averages_classifier(df_classification, selectedTW, times, TW_lengths[selectedTW])
    Study3_functions.plotting(df_mean_classifier, df_total_mean, selectedTW, TW_lengths[selectedTW], times)
    Study3_functions.global_averages(df_mean_classifier, df_total_mean, selectedTW)

elif analysing == 1: # Classification by laterality 
    ipsi_feat_lab, contra_feat_lab = Study3_functions.separate_fl_laterality(df_feature_label)
    feat_lab_lateral = [ipsi_feat_lab, contra_feat_lab]
    selectedTW = Study3_functions.tw_selection(TW_lengths)
    df_classif_lateral = Study3_functions.classify_laterality(feat_lab_lateral, TW_lengths[selectedTW], times)
    select_lateral = Study3_functions.lateral_selection()
    if select_lateral ==0: #ipsilateral
        df_classif_ipsi = df_classif_lateral[0]
        df_mean_classifier_ipsi, df_total_mean_ipsi = Study3_functions.df_averages_classifier(df_classif_ipsi, selectedTW, times,TW_lengths[selectedTW])
        Study3_functions.plotting(df_mean_classifier_ipsi, df_total_mean_ipsi, selectedTW, TW_lengths[selectedTW], times)
    elif select_lateral ==1: #contralateral
        df_classif_contra = df_classif_lateral[1]
        df_mean_classifier_contra, df_total_mean_contra = Study3_functions.df_averages_classifier(df_classif_contra, selectedTW, times,TW_lengths[selectedTW])
        Study3_functions.plotting(df_mean_classifier_contra, df_total_mean_contra, selectedTW, TW_lengths[selectedTW], times)
    elif select_lateral ==2: #comparison between ipsilateral and contralateral
        df_classif_ipsi = df_classif_lateral[0]
        df_mean_classifier_ipsi, df_total_mean_ipsi = Study3_functions.df_averages_classifier(df_classif_ipsi, selectedTW, times,TW_lengths[selectedTW])
        df_classif_contra = df_classif_lateral[1]
        df_mean_classifier_contra, df_total_mean_contra = Study3_functions.df_averages_classifier(df_classif_contra, selectedTW, times,TW_lengths[selectedTW])
        Study3_functions.plotting_lateral(df_mean_classifier_ipsi, df_total_mean_ipsi, df_mean_classifier_contra, df_total_mean_contra, selectedTW, TW_lengths[selectedTW], times)
    df_classif_ipsi = df_classif_lateral[0]
    df_mean_classifier_ipsi, df_total_mean_ipsi = Study3_functions.df_averages_classifier(df_classif_ipsi, selectedTW, times,TW_lengths[selectedTW])
    df_classif_contra = df_classif_lateral[1]
    df_mean_classifier_contra, df_total_mean_contra = Study3_functions.df_averages_classifier(df_classif_contra, selectedTW, times,TW_lengths[selectedTW])
    Study3_functions.global_averages_lateral(df_mean_classifier_ipsi, df_total_mean_ipsi, df_mean_classifier_contra, df_total_mean_contra, selectedTW)
    
elif analysing == 2: # Feature importance per electrodes (general)
    analyse_subjects = Study3_functions.analyse_feature_importance()
    subj_importances=Study3_functions.feature_importance(df_feature_label, times)
    if analyse_subjects ==0:
        selected_subject= Study3_functions.selection_subject(subjects)
        Study3_functions.filterering_plotting_feature(subj_importances[selected_subject], times)
    else:
        Study3_functions.filterering_plotting_feature_allsubj(subj_importances, times)
        
elif analysing == 3: # Feature importance per lateral
    features_labels_lateral = Study3_functions.create_features_labels_lateral(all_bins, times)
    feature_imprt_all_lateral = Study3_functions.feature_importance_laterality(features_labels_lateral, times)
    analyse_subjects = Study3_functions.analyse_feature_importance()
    if analyse_subjects == 0:
        selected_subject = Study3_functions.selection_subject(subjects)
        feature_imprt_all_lateral_selected = Study3_functions.feature_all_lateral_selection(feature_imprt_all_lateral,selected_subject)
        Study3_functions.filterering_plotting_feature_laterality(feature_imprt_all_lateral_selected, times)
    else:
        Study3_functions.filterering_plotting_feature_allsubj_laterality(feature_imprt_all_lateral, times)

elif analysing ==4: 
    Study3_functions.testing_ERPs(all_bins)