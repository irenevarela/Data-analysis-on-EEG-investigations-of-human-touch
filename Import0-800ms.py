import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle

def left_right(trialtype):
    if int(trialtype) == 1 or int(trialtype) == 4:
        return 'L'
    elif int(trialtype) == 2 or int(trialtype) == 3:
        return 'R'
    elif int(trialtype) >= 5 and int(trialtype) <=8:
        return 'X' #invalid trials
    
def attended(trialtype):
    if int(trialtype) == 1 or int(trialtype) == 2:
        return 'A'
    elif int(trialtype) == 3 or int(trialtype) == 4:
        return 'U'
    elif int(trialtype) >= 5 and int(trialtype) <=8:
        return 'X' 
    
def ipsi_contra(side, elec):
    leftElects = ['C3','F3','P3','O1']
    if ((elec in leftElects) and (side == 'L')) or ((elec not in leftElects) and (side == 'R')):
        return 'I'
    else:
        return 'C'

def binning(seq_vec,start,end,step):
    seq_vec = np.sum(seq_vec,1)
    bins = np.arange(start,end+1,step)
    bin_vec = []
    for b in range(len(bins)-1):
        bin_vec.append(sum(seq_vec[bins[b]+1:bins[b+1]]))
    return bin_vec
    
path = 'C:\Users\Irene\Documents\EEG\ExportWaveletsCTI/'

elecs = ['F3','F4','C3','C4','P3','P4','O1','O2']
leftElects = ['C3','P3','F3']
lateral = ['I','C']
conditions = ['A','B','C']
totTrials = {'A':560,'B':336,'C':560}
subjects = ['001','010','011','012','013','002','003','044','005','006','007','008','009']
attend = ['A','U']

trialtype = {}
RT = {}
for s in subjects:
    for c in conditions:
        trialtype[s+c] = []
        RT[s+c] = []
        
for trialfile in [f for f in os.listdir(path) if f.endswith('.txt')]:
    print trialfile
    cond = trialfile[:trialfile.find('.txt')].split()[0]
    n_line = 0
    trialf = io.open(path+trialfile,'r', encoding='utf-16-le')
    trialf.readline()
    for line in trialf:
        trialFields = line.split()
        subj = '{:0>3}'.format(trialFields[0])
        trialtype[subj+cond].append(trialFields[2])
        RT[subj+cond].append(trialFields[5])
    
validlist = {}
for markerfile in [f for f in os.listdir(path) if f.endswith('.MRK')]:
    print markerfile
    fileFields = markerfile[:markerfile.find('.MRK')].split('_')
    subj = fileFields[0]
    cond = fileFields[1]
    validlist[subj+cond] = []
    with open(path+markerfile,'r') as markf:
        markf.readline()
        for line in markf:
            validlist[subj+cond].append(line.find('Bad_Interval')==-1)
                  
steps = 1400
count = 1
freqsOfInterest = range(7,10) 
start_step = 500 # in 2ms; 1000ms = 500steps = stimulus (cue)
stop_step = 900 # keep 800 ms (400 points) after simulus (target)

start_bins = 0 # from 200 steps = 400ms after stimulus
end_bins = 400 # to 300 steps = 600ms after stimulus
bin_size = 5 # 10ms bins

bins = {}
rt = {}

subjects.remove('003')

for s in subjects:
    print s
    alldf = None
    for datafile in [f for f in os.listdir(path) if f.endswith('.dat')]:
        print datafile
        fileFields = datafile[:datafile.find('.dat')].split()
        subj = fileFields[0]
        cond = fileFields[1]
        elec1 = fileFields[-1][:2]
        elec2 = fileFields[-1][2:]

        if subj == s: 
            df = pd.read_csv(path+datafile, delim_whitespace=True, header=None)
            n_trials = df.shape[0]/steps
            df['subject'] = pd.Series([subj]*df.shape[0])
            df['cond'] = pd.Series([cond]*df.shape[0])
            df['trialtype'] = np.repeat(trialtype[subj+cond], steps)
            df['RT'] = np.repeat(RT[subj+cond], steps)
            df['trial'] = np.repeat(range(1,n_trials+1),steps)
            df['timestamp'] = np.tile(range(1,steps+1),n_trials)
            df['valid'] = np.repeat(validlist[subj+cond], steps)
            df['side'] = np.repeat([left_right(tt) for tt in trialtype[subj+cond]], steps)
            df['attended'] = np.repeat([attended(tt) for tt in trialtype[subj+cond]], steps)
            
            df = df[(df.timestamp > start_step -1) & (df.timestamp < stop_step)] # time filter
            
            df1 = df.loc[:, [c for c in df.columns if type(c)==type('') or int(c) in freqsOfInterest]]
            df1 = df1.reset_index()
            df1['electrode'] = pd.Series([elec1]*df1.shape[0])
            df1['lateral'] = [ipsi_contra(side,elec) for (side,elec) in zip(df1.side,df1.electrode)]
            df1=df1[(df1['side']<>'X') & (df1['valid'] == 1)]
            print df1.shape[0]
            
            df2 = df.loc[:, [c for c in df.columns if type(c)==type('') or int(c)-20 in freqsOfInterest]]
            df2 = df2.rename(columns=dict([(c, c if type(c)==type('') else int(c)-20) for c in df2.columns]))
            df2 = df2.reset_index()
            df2['electrode'] = pd.Series([elec2]*df2.shape[0])
            df2['lateral'] = [ipsi_contra(side,elec) for (side,elec) in zip(df2.side,df2.electrode)]
            df2=df2[(df2['side']<>'X') & (df2['valid'] == 1)]
            print df2.shape[0]
            
            if alldf is None:
                alldf = pd.concat((df1,df2))
            else:
                alldf = pd.concat((alldf,df1,df2))
    
            print alldf.shape[0]
            
    
    for e in elecs:
        sf = 1

        for l in lateral:
            for c in conditions:
                for a in attend:
               
                    cond_df = alldf[(alldf['subject']==s) & (alldf['cond']==c) & (alldf['electrode']==e) & (alldf['lateral']==l) & (alldf['attended']==a)]
                    bins[(s,e,c,l,a)] = []
                    rt[(s,e,c,l,a)] = []
                    for t in set(cond_df['trial'].tolist()):
                        trial_df = cond_df[cond_df['trial']==t]
                        trial = trial_df.as_matrix(columns=trial_df.columns[1:4])
                        
                        cond_bins = binning(trial,start_bins,end_bins,bin_size)
                        bins[(s,e,c,l,a)].append(cond_bins)
                        rt[(s,e,c,l,a)].append(trial_df['RT'].tolist()[0])

filename = 'BinsRT0-800.txt'
with open(filename,'wb') as fp: cPickle.dump([bins,rt],fp)

all_bins=pd.DataFrame() #dataframe alpha activations and responses - all trials
for s in subjects:
    for e in elecs:
        for c in conditions:
            for l in lateral:
                for a in attend:
                    case=bins[(s,e,c,l,a)]
                    response_time=rt[(s,e,c,l,a)]
                    for i in range(len(case)): # or range(len(r_time)), it should be the same
                        df=pd.DataFrame()
                        df1=pd.DataFrame()
                        df['subject'] = pd.Series([s])
                        df['electrode'] = pd.Series([e])
                        df['condition'] = pd.Series([c])
                        df['lateral'] = pd.Series([l])
                        df['attend'] = pd.Series([a])  
                        if ((e in leftElects) and (l == 'I')) or ((e not in leftElects) and (l == 'C')):
                            df['side'] = pd.Series(['L'])
                        else:
                            df['side'] = pd.Series(['R'])
                        df['trial'] = pd.Series([i])
                        df['rt'] = pd.Series([response_time[i]])
                        trial=case[i]
                        for k in range((np.array(case)).shape[1]):
                            df2=pd.DataFrame()
                            df2['bin'+' '+str(k)]=pd.Series(trial[k])
                            df1=pd.concat([df1,df2], axis=1)
                        df=pd.concat([df, df1], axis=1)   
                        all_bins=all_bins.append(df)
all_bins.reset_index(drop=True)
filename = 'AllBinsRT0-800'
all_bins.to_pickle(filename)

all_bins=pd.DataFrame()  #dataframe alpha activations and responses - averages per trial
for s in subjects:
    for e in elecs:
        for c in conditions:
            for l in lateral:
                for a in attend:
                    df=pd.DataFrame()
                    df1=pd.DataFrame()
                    case=bins[(s,e,c,l,a)]
                    response_time=[int(i) for i in rt[(s,e,c,l,a)]]
                    df['subject'] = pd.Series([s])
                    df['electrode'] = pd.Series([e])
                    df['condition'] = pd.Series([c])
                    df['lateral'] = pd.Series([l])
                    df['attend'] = pd.Series([a])  
                    if ((e in leftElects) and (l == 'I')) or ((e not in leftElects) and (l == 'C')):
                        df['side'] = pd.Series(['L'])
                    else:
                        df['side'] = pd.Series(['R'])
                    for i in range((np.array(case)).shape[1]):
                        df2=pd.DataFrame()
                        case_mean=np.sum((np.array(case)),axis=0)/len(case)
                        df2['bin'+' '+str(i)]=pd.Series(case_mean[i])
                        df1=pd.concat([df1, df2], axis=1)
                    df=pd.concat([df, df1], axis=1)
                    df['rt'] = pd.Series(sum(int(i) for i in response_time)/len(response_time))
                    all_bins=all_bins.append(df)       
all_bins.reset_index(drop=True) 
filename = 'AveragedAllBinsRT0-800'
all_bins.to_pickle(filename)
