#Import Data
import pickle
import numpy as np
import pandas as pd

data1 = pd.read_csv('../Data/HTAC_Qry_355.csv')
data2 = pd.read_csv('../Data/HTAC_Qry_356.csv')
data = pd.merge(data1,data2, on ='ptid')

# selection_columns
demographic_columns = [
    'ptid',
    'age',
    'gender'
]

survey_columns = [
    'asrs_score',
    'bis_factor1_ci',
    'bis_factor2_bi',
    'chaphypo_total',
    'chapinf_total',
    'chapper_total', 
    'chapphy_total',
    'chapsoc_total',
    'cvlt_totcor',
    'cvlt_bf',
    'dysfunc_pos',
    'dysfunc_neg',
    'func_pos',
    'func_neg',
    'harmavoidance',
    'hopkins_somatization',
    'hopkins_obscomp',
    'hopkins_intsensitivity',
    'hopkins_depression',
    'hopkins_anxiety',
    'novelty',
    'persistance',
    'reward_dependence',
    'scoree',
    'scorei',
    'scorev'
]

task_columns = [
    'ant_conflict_acc_effect',
    'ant_conflict_rt_effect',
    'bart_meanadjustedpumps',
    'cpt_fa',
    'cpt_hits',
    'crt_time1',
    'crt_time2',
    'ddt_total_k',
    'scap_max_capac',
    'scwt_conflict_acc_effect',
    'scwt_conflict_rt_effect',
    'smnm_main_mn',
    'smnm_main_mdrt',
    'smnm_manip_mn',
    'smnm_manip_mdrt',
    'sr_rec_explicitlearning_acc',
    'sr_rec_explicitlearning_rt',
    'sr_enc_priming_acc',
    'sr_enc_priming_rt',
    'sst_ses_ssrt_quant',
    'ts_costlong',
    'ts_costshort',
    'ts_interference',
    'vcap_max_capac',
    'vmnm_main_mn',
    'vmnm_main_mdrt',
    'vmnm_manip_mn',
    'vmnm_manip_mdrt',
]

rename_dict = {
    'scoree': 'eysenck_scoree',
    'scorei': 'eysenck_scorei',
    'scorev': 'eysenck_scorev',
    'harmavoidance': 'tci_harmavoidance',
    'novelty': 'tci_novelty',
    'persistance': 'tci_persistance',
    'reward_dependence': 'tci_reward_dependence',
    'func_pos': 'dick_func_pos',
    'func_neg': 'dick_func_neg',
    'dysfunc_pos': 'dick_dysfunc_pos',
    'dysfunc_neg': 'dick_dysfunc_neg'
}



all_columns = demographic_columns + survey_columns + task_columns
data = data.loc[:,all_columns]
data = data.replace(to_replace='.', value=np.nan).astype(float)
survey_data = data.drop(task_columns, axis = 1)
task_data = data.drop(survey_columns, axis = 1)

data = data.rename(columns = rename_dict)
task_data = task_data.rename(columns = rename_dict)
survey_data = survey_data.rename(columns = rename_dict)

data = {'all_data': data, 'survey_data': survey_data, 'task_data': task_data}
pickle.dump(data,open('../Data/subset_data.pkl','wb'))
