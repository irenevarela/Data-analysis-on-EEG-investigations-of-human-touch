import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
from scipy.stats.stats import pearsonr

elecs = ['C3','C4','P3','P4','F3','F4']
central_elecs = ['C3','C4']
leftElects = ['C3','P3','F3']
lateral = ['I','C']
conditions = ['A','B','C']
subjects = ['001','010','011','012','013','002','044','005','006','007','008','009']
attend = ['A','U']
side = ['L','R']

time_windows = [[0,200],[200,400],[400,600],[600,800]]
time_steps = 800 # in ms
bins = 800/2 # 1 bin 2 ms
bin_size = 5
number_bins = bins/bin_size

central=['C3','C4']
parietal=['P3','P4']
frontal=['F3','F4']
pairs=[central,parietal,frontal]
name_pairs=['central','parietal','frontal']
color_pairs=['red','blue','green','yellow','magenta','cyan']
def selection (df,column,param):
    df1=df.drop(df.index[[0,df.shape[0]-1]])
    for i in range(df.shape[0]):
        if param in df.iloc[i,df.columns.get_loc(column)]:
            df1=df1.append(df.iloc[i])
    return df1

def creating_tw (alldf):
    df_tw = pd.DataFrame()
    for s in subjects:
        df_subject=selection(alldf,'subject',s)
        for e in elecs:
            df_elecs=selection(df_subject,'electrode',e)
            for c in conditions:
                df_condition=selection(df_elecs,'condition',c)
                for l in lateral:
                    df_lateral=selection(df_condition,'lateral',l)
                    for a in attend:
                        df_attend=selection(df_lateral,'attend',a)
                        rt = []
                        df1 = pd.DataFrame()
                        df1['subject'] = pd.Series([s]*df_attend.shape[0])
                        df1['electrode'] = pd.Series([e]*df_attend.shape[0])
                        df1['condition'] = pd.Series([c]*df_attend.shape[0])
                        df1['lateral'] = pd.Series([l]*df_attend.shape[0])
                        df1['attend'] = pd.Series([a]*df_attend.shape[0])
                        if ((e in leftElects) and (l == 'I')) or ((e not in leftElects) and (l == 'C')):
                            df1['side'] = pd.Series(['L']*df_attend.shape[0])
                        else:
                            df1['side'] = pd.Series(['R']*df_attend.shape[0])
                        df1['trial'] = pd.Series(range(df_attend.shape[0]))
                        for t in time_windows:
                            start_bin=t[0]/10
                            end_bin=t[1]/10-1
                            mean_tw = []
                            for i in range(df_attend.shape[0]):
                                activ_tw = np.asarray(df_attend.iloc[i,df_attend.columns.get_loc('bin'+' '+str(start_bin)):df_attend.columns.get_loc('bin'+' '+str(end_bin))+1].values.tolist())
                                mean_activ_tw = np.mean(activ_tw)
                                mean_tw.append(mean_activ_tw)
                            df1[str(t[0])+' '+str(t[1])+' '+'tw'] = pd.Series(mean_tw)
                        for k in range(df_attend.shape[0]):
                            rt.append(df_attend.iloc[k,df_attend.columns.get_loc('rt')])
                        df1['rt'] = pd.Series(rt)
                        df_tw = df_tw.append(df1)   
    return df_tw

def selection_li (df,column,param):
    df1=pd.DataFrame()
    for i in range(df.shape[0]):
        if param in df.iloc[i,df.columns.get_loc(column)]:
            df1=df1.append(df.iloc[i])
    return df1

def laterality_index_df(df_tw):
    alldf1=pd.DataFrame()
    for c in conditions:
        df_cond=selection_li(df_tw,'condition',c)
        for s in subjects:
            df_subj=selection_li(df_cond,'subject',s)
            for a in attend:
                df_attend=selection_li(df_subj,'attend',a)
                for d in side:
                    df_side=selection_li(df_attend,'side',d)
                    for p_c,p in enumerate(pairs):
                        df1 = pd.DataFrame()
                        for i in range(df_side.shape[0]/len(elecs)):
                            df=pd.DataFrame()
                            df['subject'] = pd.Series([s])
                            df['condition'] = pd.Series([c])
                            df['attend'] = pd.Series([a])
                            df['side'] = pd.Series([d])
                            df['trial'] = pd.Series([i])
                            df['electrode']=pd.Series([name_pairs[p_c]])
                            df1 = df1.append(df)
                        for t_count,t in enumerate(time_windows):
                            df3 = pd.DataFrame()
                            start_tw = t[0]
                            end_tw = t[1]
                            if d=='L':
                                df_elect1=selection_li(df_side,'electrode',p[0])
                                df_elect1=selection_li(df_elect1,'lateral','I')
                                y1=np.zeros(df_elect1.shape[0])
                                y2=np.zeros(df_elect1.shape[0])
                                x=np.zeros(df_elect1.shape[0])
                                for i in range(df_elect1.shape[0]):
                                    y1[i]=df_elect1.iloc[i,df_elect1.columns.get_loc(str(start_tw)+' '+str(end_tw)+' '+'tw')]
                                    x[i]=df_elect1.iloc[i,df_elect1.columns.get_loc('rt')]
                                df_elect2=selection_li(df_side,'electrode',p[1])
                                df_elect2=selection_li(df_elect2,'lateral','C')
                                for i in range(df_elect2.shape[0]):
                                    y2[i]=df_elect2.iloc[i,df_elect2.columns.get_loc(str(start_tw)+' '+str(end_tw)+' '+'tw')]
                            else:
                                df_elect1=selection_li(df_side,'electrode',p[1])
                                df_elect1=selection_li(df_elect1,'lateral','I')
                                y1=np.zeros(df_elect1.shape[0])
                                y2=np.zeros(df_elect1.shape[0])
                                x=np.zeros(df_elect1.shape[0])
                                for i in range(df_elect1.shape[0]):
                                    y1[i]=df_elect1.iloc[i,df_elect1.columns.get_loc(str(start_tw)+' '+str(end_tw)+' '+'tw')]
                                    x[i]=df_elect1.iloc[i,df_elect1.columns.get_loc('rt')]
                                df_elect2=selection_li(df_side,'electrode',p[0])
                                df_elect2=selection_li(df_elect2,'lateral','C')
                                for i in range(df_elect2.shape[0]):
                                    y2[i]=df_elect2.iloc[i,df_elect2.columns.get_loc(str(start_tw)+' '+str(end_tw)+' '+'tw')]
                            y=y1-y2
                            y3=y1+y2
                            y_li=np.zeros(df_elect1.shape[0])
                            for i in range(len(x)):
                                y_li[i]=y[i]/y3[i]    
                            x.tolist()
                            y_li.tolist()   
                            for i in range(len(x)):
                                df2 = pd.DataFrame()
                                df2['lateral index'+' '+str(start_tw)+' '+str(end_tw)+' '+'tw'] = pd.Series([y_li[i]])
                                if t_count == 0:
                                    df2['rt'] = pd.Series([x[i]])
                                df3 = df3.append(df2)
                            df1 = pd.concat([df1, df3], axis=1)
                        alldf1=alldf1.append(df1)
    return alldf1

def tw_selection():
    for i in range(len(time_windows)):
        print (str(i)+':'+' '+str(time_windows[i][0])+' '+str(time_windows[i][1])+' '+'tw')
    selected=input()
    return time_windows[selected]

def selection_analysis():
    print('\n Analysis: \n 0: ERPs vs. RT \n 1: Laterality Index vs. RT \n 2: Test correlation ERPs vs. RT \n 3: Test correlation LI vs. RT \n 4: RTs attended vs. unattended' )
    selected=input()
    return selected

def selection_ERP_RT():
    print ('Scatter options: \n 0: Without filtering \n 1: Filter by attended/unattended \n 2: Filter by electrode')
    selected=input()
    return selected

def scatter_plot(df, title, task, start, end, position):
    x = []
    x_int = []
    y = []
    for i in range(df.shape[0]):
        x.append(df.iloc[i,df.columns.get_loc('rt')])
        y.append(df.iloc[i,df.columns.get_loc(str(start)+' '+str(end)+' '+'tw')])
    for k in x:
        x_int.append(int(k))
    plt.subplot(2,3,position)
    plt.scatter(x_int, y, s=0.5, color='b')
    plt.xlabel('Response Times (ms)')
    plt.ylabel('Alpha Activation')
    plt.title(title+' '+str(task)+' '+'task')
    plt.subplots_adjust(left=0.09, right=0.95, top=0.915, bottom=0.1, hspace=0.335, wspace=0.29)
    plt.show()

def scatter_plot_attended (df, title, task, start, end, position):
    plt.subplot(2,3,position)
    for count, a in enumerate(attend):
        x = []
        y = []
        df_attend=selection_li(df,'attend',a)
        for i in range(df_attend.shape[0]):
            x.append(df_attend.iloc[i,df_attend.columns.get_loc('rt')])
            y.append(df_attend.iloc[i,df_attend.columns.get_loc(str(start)+' '+str(end)+' '+'tw')])
        x_int = []
        for k in x:
            x_int.append(int(k))
        plt.scatter(x_int, y, s=0.4, c=color_pairs[count])
    plt.xlabel('Response Times (ms)')
    plt.ylabel('Alpha Activation')
    plt.title(title+' '+str(task)+' '+'task')
    plt.subplots_adjust(left=0.09, right=0.95, top=0.915, bottom=0.1, hspace=0.335, wspace=0.29)
    plt.show()

def scatter_plot_elecs (df, title, task, start, end, position):
    plt.subplot(2,3,position)
    for count, e in enumerate(elecs):
        x = []
        y = []
        df_elec=selection_li(df,'electrode',e)
        for i in range(df_elec.shape[0]):
            x.append(df_elec.iloc[i,df_elec.columns.get_loc('rt')])
            y.append(df_elec.iloc[i,df_elec.columns.get_loc(str(start)+' '+str(end)+' '+'tw')])
        x_int = []
        for k in x:
            x_int.append(int(k))
        plt.scatter(x_int, y, s=0.4, c=color_pairs[count])
    plt.xlabel('Response Times (ms)')
    plt.ylabel('Alpha Activation')
    plt.title(title+' '+str(task)+' '+'task')
    plt.subplots_adjust(left=0.09, right=0.95, top=0.915, bottom=0.1, hspace=0.335, wspace=0.29)
    plt.show()

def scatter_lateral_task(df_tw,selected_tw, select_ERP_RT):
    counter = 1
    start_tw = selected_tw[0]
    end_tw = selected_tw[1]
    for l in lateral:
        if l == 'I':
            title = 'Ipsilateral'
        else:
            title = 'Contralateral'
        df_lat = selection_li(df_tw,'lateral',l)
        for c in conditions:
            df_cond = selection_li(df_lat,'condition',c)
            if select_ERP_RT == 0:
                scatter_plot(df_cond, title, c, start_tw, end_tw, counter)
            elif select_ERP_RT == 1:
                scatter_plot_attended(df_cond, title, c, start_tw, end_tw, counter)
            elif select_ERP_RT == 2:
                scatter_plot_elecs(df_cond, title, c, start_tw, end_tw, counter)
            counter +=1

def selection_LI_RT():
    print ('Scatter options: \n 0: Without filtering \n 1: Filter by attended/unattended \n 2: Filter by electrode')
    selected=input()
    return selected

def plot(x,y,position,col):
    plt.subplot(1,3,position)
    plt.scatter(x, y, s=0.5, c=col)
    plt.xlabel("Response Times")
    plt.ylim(-100,100)
    plt.ylabel("Lateralization index")
    plt.show()

def scatter_latindex(df_latindex, selec_tw, select_LI_RT):
    position=1
    start_tw = selec_tw[0]
    end_tw = selec_tw[1]
    for c in conditions:
        df_cond=selection(df_latindex,'condition',c)
        for s in subjects:
            df_subj=selection(df_cond,'subject',s)
            if select_LI_RT == 0:
                count = 1
                for a in attend:
                    df_attend=selection(df_subj,'attend',a)
                    for d in side:
                        df_side=selection(df_attend,'side',d)
                        for z in name_pairs:
                            df_latind=selection(df_side,'electrode',z)
                            y=df_latind['lateral index'+' '+str(start_tw)+' '+str(end_tw)+' '+'tw'].tolist()
                            x=df_latind['rt'].tolist()
                            plot(x,y,position,color_pairs[count])
            elif select_LI_RT == 1:
                for count,a in enumerate(attend):
                    df_attend=selection(df_subj,'attend',a)
                    for d in side:
                        df_side=selection(df_attend,'side',d)
                        for z in name_pairs:
                            df_latind=selection(df_side,'electrode',z)
                            y=df_latind['lateral index'+' '+str(start_tw)+' '+str(end_tw)+' '+'tw'].tolist()
                            x=df_latind['rt'].tolist()
                            plot(x,y,position,color_pairs[count])
            elif select_LI_RT == 2:
                for a in attend:
                    df_attend=selection(df_subj,'attend',a)
                    for d in side:
                        df_side=selection(df_attend,'side',d)
                        for count,z in enumerate(name_pairs):
                            df_latind=selection(df_side,'electrode',z)
                            y=df_latind['lateral index'+' '+str(start_tw)+' '+str(end_tw)+' '+'tw'].tolist()
                            x=df_latind['rt'].tolist()
                            plot(x,y,position,color_pairs[count])
        position+=1
    return

def selection_test_type():
    print ('Test type: \n 0: Pearson linear correlation \n 1: Polynomial fitting correlation')
    selected=input()
    return selected

def polyfit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    ssres = np.sum((y - p(x))**2)
    sstot = np.sum((y - np.mean(y))**2) 
    return 1 - ssres / sstot

def test_correlation(df_bins_rt, selec_tw, select_test):
    Rcoeff = {}
    Rcoeff2 = {}
    ipsi_matr_R = np.zeros((len(elecs),len(conditions)))
    contra_matr_R = np.zeros((len(elecs),len(conditions)))
    ipsi_matr_R2 = np.zeros((len(elecs),len(conditions)))
    contra_matr_R2 = np.zeros((len(elecs),len(conditions)))
    start_bins = selec_tw[0]/10
    end_bins = selec_tw[1]/10-1
    for s in subjects:
        df_subj=selection_li(df_bins_rt,'subject',s)
        for l in lateral:
            df_lat=selection_li(df_subj,'lateral',l)
            for a in attend:
                df_attend=selection_li(df_lat,'attend',a)
                for nc, c in enumerate(conditions):
                    df_cond=selection_li(df_attend,'condition',c)
                    for ne, e in enumerate(elecs):
                        df_elec=selection_li(df_cond,'electrode',e)
                        Rcoeff[(s,e,c,l,a)] = []
                        Rcoeff2[(s,e,c,l,a)] = []
                        for b in range(start_bins, end_bins+1,1):
                            x = np.asarray(df_elec['bin'+' '+str(b)])
                            y = df_elec['rt'].values.tolist()
                            y_i = []            
                            for i in y:
                               y_i.append(int(i))
                            y_i = np.asarray(y_i)
                            R = pearsonr(x,y_i)
                            Rcoeff[(s,e,c,l,a)].append(R)
                            R2 = polyfit(x,y_i, 2)
                            Rcoeff2[(s,e,c,l,a)].append(R2)
                            if l == 'I':
                                ipsi_matr_R[ne,nc] += R[0]
                                ipsi_matr_R2[ne,nc] += R2
                            else:
                                contra_matr_R[ne,nc] += R[0]
                                contra_matr_R2[ne,nc] += R2
    ipsi_matr_R = ipsi_matr_R/(((end_bins-start_bins)+1)*len(subjects)*len(attend))
    contra_matr_R = contra_matr_R/(((end_bins-start_bins)+1)*len(subjects)*len(attend))
    ipsi_matr_R2 = ipsi_matr_R2/(((end_bins-start_bins)+1)*len(subjects)*len(attend))
    contra_matr_R2 = contra_matr_R2/(((end_bins-start_bins)+1)*len(subjects)*len(attend))
    top = max(np.max(ipsi_matr_R),np.max(contra_matr_R))
    top2 = max(np.max(ipsi_matr_R2),np.max(contra_matr_R2))
    if select_test == 0:
        plt.figure()
        ax = plt.subplot(1,2,1)
        plt.pcolor(ipsi_matr_R, cmap='bwr', vmin=-top, vmax=top)
        plt.title('Ipsilateral RT correlation mean')
        plt.xlabel('conditions')
        plt.ylabel('electrodes')
        ax.set_yticks(np.arange(len(elecs)) + 0.5, minor=False)
        ax.set_yticklabels(elecs)
        ax.set_xticks(np.arange(len(conditions)) + 0.5, minor=False)
        ax.set_xticklabels(conditions)
        plt.colorbar()
        ax = plt.subplot(1,2,2)
        plt.pcolor(contra_matr_R, cmap='bwr', vmin=-top, vmax=top)
        plt.title('Contralateral RT correlation mean')
        plt.xlabel('conditions')
        plt.ylabel('electrodes')
        ax.set_yticks(np.arange(len(elecs)) + 0.5, minor=False)
        ax.set_yticklabels(elecs, minor=False)
        ax.set_xticks(np.arange(len(conditions)) + 0.5)
        ax.set_xticklabels(conditions, minor=False)
        plt.colorbar()
    elif select_test == 1:
        plt.figure()
        ax = plt.subplot(1,2,1)
        plt.pcolor(ipsi_matr_R2, cmap='bwr', vmin=-top2, vmax=top2)
        plt.title('Ipsilateral RT correlation mean')
        plt.xlabel('conditions')
        plt.ylabel('electrodes')
        ax.set_yticks(np.arange(len(elecs)) + 0.5, minor=False)
        ax.set_yticklabels(elecs)
        ax.set_xticks(np.arange(len(conditions)) + 0.5, minor=False)
        ax.set_xticklabels(conditions)
        plt.colorbar()
        ax = plt.subplot(1,2,2)
        plt.pcolor(contra_matr_R2, cmap='bwr', vmin=-top2, vmax=top2)
        plt.title('Contralateral RT correlation mean')
        plt.xlabel('conditions')
        plt.ylabel('electrodes')
        ax.set_yticks(np.arange(len(elecs)) + 0.5, minor=False)
        ax.set_yticklabels(elecs, minor=False)
        ax.set_xticks(np.arange(len(conditions)) + 0.5)
        ax.set_xticklabels(conditions, minor=False)
        plt.colorbar()

def test_correlation_li(df_laterality_indx, selec_tw):
    start_tw = selec_tw[0]
    end_tw = selec_tw[1]
    pearson_R = np.zeros((len(name_pairs),len(conditions)))
    fitting_R2 = np.zeros((len(name_pairs),len(conditions)))
    for s in subjects:
        df_sub=selection_li(df_laterality_indx,'subject',s)
        for a in attend:
            df_att=selection_li(df_sub,'attend',a)
            for d in side:
                df_side=selection_li(df_att,'side',d)
                for nc, c in enumerate(conditions):
                    df_cond=selection_li(df_side,'condition',c)
                    for ne, e in enumerate(name_pairs):
                        df_elecs=selection_li(df_cond,'electrode',e)
                        x = np.asarray(df_elecs['lateral index'+' '+str(start_tw)+' '+str(end_tw)+' '+'tw'].tolist())
                        y = np.asarray(df_elecs['rt'].tolist())
                        R=pearsonr(x,y)
                        R2 = polyfit(x,y,2)
                        pearson_R[ne,nc] += R[0]
                        fitting_R2[ne,nc] += R2
    pearson_R = pearson_R/((df_elecs.shape[0])*len(subjects)*len(attend)*len(side))
    fitting_R2 = fitting_R2/((df_elecs.shape[0])*len(subjects)*len(attend)*len(side))
    top = max(np.max(pearson_R),np.max(fitting_R2))
    plt.figure()
    ax = plt.subplot(1,2,1)
    plt.pcolor(pearson_R, cmap='bwr', vmin=-top, vmax=top)
    plt.title('Lateral Index RT correlation - Pearson')
    plt.xlabel('conditions')
    plt.ylabel('electrode pairs')
    ax.set_yticks(np.arange(len(name_pairs)) + 0.5, minor=False)
    ax.set_yticklabels(name_pairs)
    ax.set_xticks(np.arange(len(conditions)) + 0.5, minor=False)
    ax.set_xticklabels(conditions)
    plt.colorbar()
    ax = plt.subplot(1,2,2)
    plt.pcolor(fitting_R2, cmap='bwr', vmin=-top, vmax=top)
    plt.title('Lateral Index RT correlation - Fitting')
    plt.xlabel('conditions')
    plt.ylabel('electrode pairs')
    ax.set_yticks(np.arange(len(name_pairs)) + 0.5, minor=False)
    ax.set_yticklabels(name_pairs, minor=False)
    ax.set_xticks(np.arange(len(conditions)) + 0.5)
    ax.set_xticklabels(conditions, minor=False)
    plt.colorbar()

def rt_attention(df_tw):
    responses = []
    for a in attend:
        df_attend=selection_li(df_tw,'attend',a)
        for c in conditions:
            x=[]
            df_cond=selection_li(df_attend,'condition',c)
            for i in range(df_cond.shape[0]):
                x.append(df_cond.iloc[i,df_cond.columns.get_loc('rt')])
            x_int = []
            for k in x:
                x_int.append(int(k))
            responses.append(np.mean(np.asarray(x_int)))
    rt_attend = responses[0:3]
    rt_unattend = responses[3:6]
    x = np.arange(len(rt_attend))
    plt.bar(x-0.2, rt_attend, color='r', width=0.2)
    plt.bar(x, rt_unattend, color='b', width=0.2)
    plt.title('Response time comparison between attended and unattended trials')
    plt.xticks(np.arange(len(conditions)), conditions)
    plt.ylabel('Response time (ms)')
    plt.xlabel('Type of task')
