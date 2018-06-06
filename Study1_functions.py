import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
import itertools

elecs = ['C3','C4','P3','P4','F3','F4'] 
leftElects = ['C3','P3','F3']
conditions = ['A','B','C']
subjects = ['001','010','011','012','013','002','044','005','006','007','008','009']
lateral = ['I','C']
attend = ['A','U']
side = ['L','R']

time_steps = 800 # in ms
bins = 800/2 # 1 bin 2 ms
bin_size = 5
number_bins = bins/bin_size
start_time = 300 #ms
end_time = 600 #ms

def analyse_consistency():
    print ('Analysis options: \n 0: Ipsilateral vs. Contralateral \n 1: Left vs. Right \n 2: Attended vs. Unattended')
    selected=input()
    return selected

def df_parameters_choosing (df,column,param):
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    for i in range(df.shape[0]):
        if param in df.iloc[i,df.columns.get_loc(column)]:
            df1=df1.append(df.iloc[i])
    return df1

def selection_bins_averages(alldf):
    df = pd.DataFrame()
    for s in subjects:
        df_subject=df_parameters_choosing(alldf,'subject',s)
        for e in elecs:
            df_elecs=df_parameters_choosing(df_subject,'electrode',e)
            for c in conditions:
                df_condition=df_parameters_choosing(df_elecs,'condition',c)
                for l in lateral:
                    df_lateral=df_parameters_choosing(df_condition,'lateral',l)
                    for a in attend:
                        df_attend=df_parameters_choosing(df_lateral,'attend',a)
                        df1 = pd.DataFrame()
                        df1['subject'] = pd.Series([s])
                        df1['electrode'] = pd.Series([e])
                        df1['condition'] = pd.Series([c])
                        df1['lateral'] = pd.Series([l])
                        df1['attend'] = pd.Series([a])
                        if ((e in leftElects) and (l == 'I')) or ((e not in leftElects) and (l == 'C')):
                            df1['side'] = pd.Series(['L'])
                        else:
                            df1['side'] = pd.Series(['R'])
                        for i in range(number_bins):
                            activations_per_trial = np.asarray(df_attend['bin'+' '+str(i)])
                            mean_activations = np.mean(activations_per_trial)
                            df1['bin'+' '+str(i)] = pd.Series([mean_activations])
                        rt_per_trial = df_attend['rt'].values.tolist()
                        rt_trial = []
                        for i in range(len(rt_per_trial)):
                            rt_trial.append(int(rt_per_trial[0]))
                        mean_rt = np.mean(np.asarray(rt_trial))
                        df1['rt'] = pd.Series([mean_rt])
                        df=df.append(df1)
    return df

def variable_selection (selection):
    for j in range(len(selection)):
        print(str(j)+":"+str(selection[j]))  
    select_analyse=raw_input()
    if type(select_analyse)!= int:
        input_list = select_analyse.split(',')    
    return input_list

def variables_in (variable):
    user_input = []
    if 'subject' in variable:
        user_input.append(raw_input("Choose subjects:"+" "+str(subjects)+"\n").split(','))
    if 'electrode' in variable:
        user_input.append(raw_input("Choose electrodes:"+" "+str(elecs)+"\n").split(','))
    if 'condition' in variable:
        user_input.append(raw_input("Choose condition:"+" "+str(conditions)+"\n").split(','))
    if 'lateral' in variable:
        user_input.append(raw_input("Choose lateral:"+" "+str(lateral)+"\n"))
    if 'attend' in variable:
        user_input.append(raw_input("Choose attend:"+" "+str(attend)+"\n"))
    if 'side' in variable:
        user_input.append(raw_input("Choose side:"+" "+str(side)+"\n"))
    return user_input

def select_df (df,selection,variable):
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    for s in selection:
        for e in s:
            for i in range(df.shape[0]):
                if e in df.iloc[i,df.columns.get_loc(variable[selection.index(s)])]:
                    df1=df1.append(df.iloc[i])
        df=df1
        df1=df.drop(df.index[[0,df.shape[0]-1]])
    return df

def combination (variable,selection):
    if len(variable)==2:
        combinations=list(itertools.product(selection[0],selection[1]))
    elif len(variable)==3:
        combinations=list(itertools.product(selection[0],selection[1],selection[2]))
    elif len(variable)==4:
        combinations=list(itertools.product(selection[0],selection[1],selection[2],selection[3]))
    return combinations
   
def mean_value (df, combinations, variable):
    mean_combine =[]    
    for s in combinations:
        df1 = df
        for e in s:
            df2 = df.drop(df.index[[0,df.shape[0]-1]])
            for i in range(df1.shape[0]):
                if e in df1.iloc[i,df1.columns.get_loc(variable[s.index(e)])]:
                    df2 = df2.append(df1.iloc[i])
            df1 = df2  
        df3 = df.drop(df.index[[0,df.shape[0]-1]])
        for d in range(df2.shape[0]):
            df3 = df3.append(df2.iloc[d,df2.columns.get_loc('bin'+' '+str(start_time/10)):df2.columns.get_loc('bin'+' '+str(end_time/10-1))+1])
        mean_combine.append(np.sum(df3.iloc[:,df3.columns.get_loc('bin'+' '+str(start_time/10)):df3.columns.get_loc('bin'+' '+str(end_time/10-1))+1].as_matrix(), axis=0)/df3.shape[0]) 
    return mean_combine

def plotting (mean,title,combinations,ax):
    plt.pcolor(mean, cmap='coolwarm', vmin=round(np.amin(mean)-1), vmax=round(np.amax(mean)+1))
    plt.title(str(title))
    plt.xlabel('time (ms)')
    ax.set_yticks(np.arange(len(combinations)) + 0.5, minor=False)
    ax.set_yticklabels(combinations, minor=False, fontsize=8)
    ax.set_xticks(np.arange(len(mean[0])+1))
    ax.set_xticklabels(np.arange(start_time,end_time+1,10), fontsize=7.5)
    plt.colorbar(orientation='horizontal')
  
def plot_activations_not_lin_comb(activations):
    tasks = ['Endogenous predictive','Exogenous', 'Endogenous counter-predictive']
    time = range(start_time, end_time,10)
    plt.figure()
    for i, activ in enumerate(activations):
        if i ==0:
            plt.subplot(1,2,1)
        else:
            plt.subplot(1,2,2)
        for k in range(len(activ)):
            sequence_activ = activ[k]
            plt.plot(time,sequence_activ)
        plt.xlabel('time (ms)')
        plt.ylabel('alpha activations amplitude')
        plt.legend(tasks, loc='upper left')
        plt.show()
      
def linear_combination_ipsi_contra (df):
    ipsi_lin_comb = []
    contra_lin_comb = []
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    activations = []
    for l in lateral:
        activations_lat = []
        A_activ = np.zeros((end_time-start_time)/10)
        B_activ = np.zeros((end_time-start_time)/10)
        C_activ = np.zeros((end_time-start_time)/10)
        for e in elecs:
            for c in conditions:
                select = [e,c,l]
                variab = ['electrode','condition','lateral']
                df_select = df
                for j in select:
                    for i in range(df_select.shape[0]):
                        if j in df_select.iloc[i,df_select.columns.get_loc(variab[select.index(j)])]:
                            df1=df1.append(df_select.iloc[i])
                    df_select=df1
                    df1=df.drop(df.index[[0,df.shape[0]-1]])  
                if c == 'A':
                    act_A = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    A_activ +=act_A
                elif c == 'B':
                    act_B = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    B_activ +=act_B
                elif c == 'C':
                    act_C = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    C_activ +=act_C
            lin_comb=abs(act_A-act_B-act_C)/abs(act_A+act_B+act_C)
            if l == 'I':
                ipsi_lin_comb.append(lin_comb)                
            else:
                contra_lin_comb.append(lin_comb)
        activations_lat.append(A_activ)
        activations_lat.append(B_activ)
        activations_lat.append(C_activ)
        activations.append(activations_lat)                  
    return ipsi_lin_comb, contra_lin_comb, lin_comb, activations

def linear_combination_left_right(df):
    right_lin_comb = []
    left_lin_comb = []
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    activations = []
    for s in side:
        activations_side = []
        A_activ = np.zeros((end_time-start_time)/10)
        B_activ = np.zeros((end_time-start_time)/10)
        C_activ = np.zeros((end_time-start_time)/10)
        for e in elecs:
            for c in conditions:
                select = [e,c,s]
                variab = ['electrode','condition','side']
                df_select = df
                for j in select:
                    for i in range(df_select.shape[0]):
                        if j in df_select.iloc[i,df_select.columns.get_loc(variab[select.index(j)])]:
                            df1=df1.append(df_select.iloc[i])
                    df_select=df1
                    df1=df.drop(df.index[[0,df.shape[0]-1]])
                if c == 'A':
                    act_A = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    A_activ +=act_A
                elif c == 'B':
                    act_B = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    B_activ +=act_B
                elif c == 'C':
                    act_C = np.sum((df_select.iloc[:,df_select.columns.get_loc('bin'+' '+str(start_time/10)):df_select.columns.get_loc('bin'+' '+str(end_time/10-1))+1]).as_matrix(), axis=0)/df_select.shape[0]
                    C_activ +=act_C
            lin_comb=abs(act_A-act_B-act_C)/abs(act_A+act_B+act_C)
            if s == 'R':
                right_lin_comb.append(lin_comb)                
            else:
                left_lin_comb.append(lin_comb) 
        activations_side.append(A_activ)
        activations_side.append(B_activ)
        activations_side.append(C_activ)
        activations.append(activations_side) 
    return right_lin_comb, left_lin_comb, lin_comb, activations

def plotting_lin_comb(type_lc, title,ax, lin_comb):
    plt.pcolor(type_lc, cmap='Reds_r', vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('time intervals')
    plt.ylabel('electrodes')
    ax.set_yticks(np.arange(len(elecs)) + 0.5, minor=False)
    ax.set_yticklabels(elecs, minor=False, fontsize=8)
    ax.set_xticks(np.arange(len(lin_comb)+1))
    ax.set_xticklabels(np.arange(start_time,end_time+1,10), fontsize=7.5)
    plt.colorbar(orientation='horizontal')