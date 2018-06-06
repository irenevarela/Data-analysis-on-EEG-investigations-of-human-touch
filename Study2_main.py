import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import cPickle
import Study2_functions

path='C:\Users\Irene\Documents\EEG/'
all_bins =cPickle.load(open('AllBinsRT0-800','rb'))
df_tw = Study2_functions.creating_tw(all_bins)
df_laterality_indx = Study2_functions.laterality_index_df(df_tw)
selec_tw = Study2_functions.tw_selection()
selection_analysis = Study2_functions.selection_analysis()
if selection_analysis ==0:
    select_ERP_RT = Study2_functions.selection_ERP_RT()
    Study2_functions.scatter_lateral_task(df_tw, selec_tw, select_ERP_RT)
elif selection_analysis ==1:
    select_LI_RT = Study2_functions.selection_LI_RT()
    Study2_functions.scatter_latindex(df_laterality_indx, selec_tw, select_LI_RT)
elif selection_analysis ==2:
    select_test = Study2_functions.selection_test_type()
    Study2_functions.test_correlation(all_bins, selec_tw, select_test)
elif selection_analysis ==3:
    Study2_functions.test_correlation_li(df_laterality_indx, selec_tw)
elif selection_analysis ==4:
    Study2_functions.rt_attention(df_tw)
    