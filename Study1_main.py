import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
import Study1_functions

path='C:\Users\Irene\Documents\EEG/'

filename = 'AveragedAllBinsRT0-800'
df = cPickle.load(open(filename,'rb'))

analysing = Study1_functions.analyse_consistency()

if analysing == 0: #Ipsilateral vs. Contralateral
    selection = ['subject', 'electrode', 'condition', 'attend', 'side']
    input_list = Study1_functions.variable_selection(selection)
    variable = []
    if str(0) in input_list:
        variable.append('subject')
    if str(1) in input_list:
        variable.append('electrode')
    if str(2) in input_list:
        variable.append('condition')
    if str(3) in input_list:
        variable.append('attend')
    if str(4) in input_list:
        variable.append('side')
    variab_values = Study1_functions.variables_in(variable)
    variable.append('lateral')# Lateral analysis
    title_ipsi = 'Ipsilateral'
    title_contra = 'Contralateral'
    variab_values_ipsi = variab_values[:]
    variab_values_ipsi.append(list('I'))
    ipsi_df = Study1_functions.select_df(df, variab_values_ipsi, variable)
    combinations_ipsi = Study1_functions.combination(variable,variab_values_ipsi)
    mean_ipsi = Study1_functions.mean_value(ipsi_df,combinations_ipsi,variable)
    ax_ipsi = plt.subplot(1,2,1)
    Study1_functions.plotting(mean_ipsi,title_ipsi,combinations_ipsi,ax_ipsi)
    variab_values_contra = variab_values[:]
    variab_values_contra.append(list('C'))
    contra_df = Study1_functions.select_df(df, variab_values_contra, variable)
    combinations_contra = Study1_functions.combination(variable,variab_values_contra)
    mean_contra = Study1_functions.mean_value(contra_df,combinations_contra,variable)
    ax_contra = plt.subplot(1,2,2)
    Study1_functions.plotting(mean_contra,title_contra,combinations_contra,ax_contra)
    plt.figure()
    ipsi_lc, contra_lc, linear_comb, activations = Study1_functions.linear_combination_ipsi_contra(df)
    ax_ipsi = plt.subplot(1,2,1)
    Study1_functions.plotting_lin_comb(ipsi_lc,title_ipsi,ax_ipsi, linear_comb)
    ax_contra = plt.subplot(1,2,2)
    Study1_functions.plotting_lin_comb(contra_lc,title_contra,ax_contra, linear_comb)
    Study1_functions.plot_activations_not_lin_comb(activations)
    
elif analysing ==1: #Left vs. Right
    selection = ['subject', 'electrode', 'condition', 'lateral', 'attend']
    input_list = Study1_functions.variable_selection(selection)
    variable = []
    if str(0) in input_list:
        variable.append('subject')
    if str(1) in input_list:
        variable.append('electrode')
    if str(2) in input_list:
        variable.append('condition')
    if str(3) in input_list:
        variable.append('lateral')
    if str(4) in input_list:
        variable.append('attend')
    variab_values = Study1_functions.variables_in(variable)
    variable.append('side') #Side analysis
    title_left = 'Left'
    title_right = 'Right'
    variab_values_left = variab_values[:]
    variab_values_left.append(list('L'))
    left_df = Study1_functions.select_df(df, variab_values_left, variable)
    combinations_left = Study1_functions.combination(variable,variab_values_left)
    mean_left = Study1_functions.mean_value(left_df,combinations_left,variable)
    ax_left = plt.subplot(1,2,1)
    Study1_functions.plotting(mean_left,title_left,combinations_left,ax_left)
    variab_values_right = variab_values[:]
    variab_values_right.append(list('R'))
    right_df = Study1_functions.select_df(df, variab_values_right, variable)
    combinations_right = Study1_functions.combination(variable,variab_values_right)
    mean_right = Study1_functions.mean_value(right_df,combinations_right,variable)
    ax_right = plt.subplot(1,2,2)
    Study1_functions.plotting(mean_right,title_right,combinations_right,ax_right)
    plt.figure()
    left_lc, right_lc, linear_comb, activations = Study1_functions.linear_combination_left_right(df)
    ax_left = plt.subplot(1,2,1)
    Study1_functions.plotting_lin_comb(left_lc,title_left,ax_left, linear_comb)
    ax_right = plt.subplot(1,2,2)
    Study1_functions.plotting_lin_comb(right_lc,title_right,ax_right, linear_comb)
    Study1_functions.plot_activations_not_lin_comb(activations)
    
elif analysing ==2: #Attended vs. Unattends
    selection = ['subject', 'electrode', 'condition', 'lateral', 'side']
    input_list = Study1_functions.variable_selection(selection)
    variable = []
    if str(0) in input_list:
        variable.append('subject')
    if str(1) in input_list:
        variable.append('electrode')
    if str(2) in input_list:
        variable.append('condition')
    if str(3) in input_list:
        variable.append('lateral')
    if str(4) in input_list:
        variable.append('side')
    variab_values = Study1_functions.variables_in(variable)
    variable.append('attend') # Attend analysis
    title_attend = 'Attended'
    title_unattend = 'Unattended'
    variab_values_attend = variab_values[:]
    variab_values_attend.append(list('A'))
    attend_df = Study1_functions.select_df(df, variab_values_attend, variable)
    combinations_attend = Study1_functions.combination(variable,variab_values_attend)
    mean_attend = Study1_functions.mean_value(attend_df,combinations_attend,variable)
    ax_attend = plt.subplot(1,2,1)
    Study1_functions.plotting(mean_attend,title_attend,combinations_attend,ax_attend)
    variab_values_unattend = variab_values[:]
    variab_values_unattend.append(list('U'))
    unattend_df = Study1_functions.select_df(df, variab_values_unattend, variable)
    combinations_unattend = Study1_functions.combination(variable,variab_values_unattend)
    mean_unattend = Study1_functions.mean_value(unattend_df,combinations_unattend,variable)
    ax_unattend = plt.subplot(1,2,2)
    Study1_functions.plotting(mean_unattend,title_unattend,combinations_unattend,ax_unattend)