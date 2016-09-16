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
    'dysfunc_total',
    'func_total',
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

rename_dict = {}
rename_dict.update({s:'S.' + s for s in survey_columns})
rename_dict.update({t:'T.' + t for t in task_columns})
rename_dict.update({
    'scoree': 'S.eysenck_scoree',
    'scorei': 'S.eysenck_scorei',
    'scorev': 'S.eysenck_scorev',
    'harmavoidance': 'S.tci_harmavoidance',
    'novelty': 'S.tci_novelty',
    'persistance': 'S.tci_persistance',
    'reward_dependence': 'S.tci_reward_dependence',
    'func_total': 'S.dick_func_total',
    'dysfunc_total': 'S.dick_dysfunc_total',
})

verbose_dict= {
    'S.asrs_score': 'Adult ADHD Scale' ,
    'S.bis_factor1_ci': 'BIS_cognitive_impulsivity',
    'S.bis_factor2_bi': 'BIS_behavioral_impulsivity',
    'S.chaphypo_total': 'Chapman hypomania',
    'S.chapinf_total': 'Chapman infrequency',
    'S.chapper_total': 'Chapman perceptual abberation', 
    'S.chapphy_total': 'Chapman physical anhedonia',
    'S.chapsoc_total': 'Chapman social anhedonia',
    'S.cvlt_totcor': 'Verbal Learning Test recall trials 1-5',
    'S.cvlt_bf':  'Verbal Learning Test interference list recall',
    'S.dick_dysfunc_total': 'Dickman dysfunctional impulsivity',
    'S.dick_func_total': 'Dickman functional impulsivity', 
    'S.tci_harmavoidance': 'Temperament and Character Inventory: Harm Avoidance',
    'S.hopkins_somatization': 'Hopkins Somatization',
    'S.hopkins_obscomp': 'Hopkins Obsessive compulsive',
    'S.hopkins_intsensitivity': 'Hopkins Intsensivitiy',
    'S.hopkins_depression': 'Hopkins Depression',
    'S.hopkins_anxiety': 'Hopkins Anxiety',
    'S.tci_novelty': 'Temperament and Character Inventory: Novely Seeking',
    'S.tci_persistance': 'Temperament and Character Inventory: Persistence',
    'S.tci_reward_dependence': 'Temperament and Character Inventory: Reward Dependence',
    'S.eysenck_scoree': 'Eysenck: Empathy',
    'S.eysenck_scorei': 'Eysenck: Impulsivity',
    'S.eysenck_scorev': 'Eysenck: Venturesome',
    'T.ant_conflict_acc_effect': 'ANT - Conflict Accuracy effect',
    'T.ant_conflict_rt_effect': 'ANT - Conflict RT effect',
    'T.bart_meanadjustedpumps': 'BART - pumps',
    'T.cpt_fa': 'CPT - False Alarms',
    'T.cpt_hits': 'CPT - hits',
    'T.cpt_dprime': "CPT d'. P(Hits)-P(FA)",
    'T.crt_time1': 'Color trails time 1: Speed of processing',
    'T.crt_time2': 'Color trails time 2: Speed of processing',
    'T.ddt_total_k': 'DDT discount rate',
    'T.scap_max_capac': 'Spatial WM capacity',
    'T.scwt_conflict_acc_effect': 'Stroop accuracy effect' ,
    'T.scwt_conflict_rt_effect': 'Stroop RT effect',
    'T.smnm_main_mn': 'Spatial Memory and Manipulation: Maintenance accuracy',
    'T.smnm_main_mdrt': 'SpatialMemory and Manipulation: Maintenance RT',
    'T.smnm_manip_mn': 'Spatial Memory and Manipulation: Manipulation accuracy',
    'T.smnm_manip_mdrt': 'Spatial Memory and Manipulation: Manipulation RT',
    'T.sr_rec_explicitlearning_acc': 'Scene recognition: Explicit learning accuracy',
    'T.sr_rec_explicitlearning_rt': 'Scene recognition: Explicit learning rt',
    'T.sr_enc_priming_acc': 'Scene recognition: implicit learning accuracy',
    'T.sr_enc_priming_rt': 'Scene recognition: implicit learning rt',
    'T.sst_ses_ssrt_quant': 'Stop signal SSRT',
    'T.ts_costlong': 'Task switch Cost long interval',
    'T.ts_costshort': 'Task switch Cost short interval',
    'T.ts_interference': 'TS conflict effect',
    'T.vcap_max_capac': 'Verbal WM capacity',
    'T.vmnm_main_mn': 'Verbal Memory and Manipulation: Maintenance accuracy',
    'T.vmnm_main_mdrt': 'Verbal Memory and Manipulation: Maintenance RT',
    'T.vmnm_manip_mn': 'Verbal Memory and Manipulation: Manipulation accuracy',
    'T.vmnm_manip_mdrt': 'Verbal Memory and Manipulation: Manipulation RT'
}


all_columns = demographic_columns + survey_columns + task_columns
data = data.loc[:,all_columns]
data = data.replace(to_replace='.', value=np.nan).astype(float)

# compute additional variables or flip sign
data.loc[:,'T.cpt_dprime'] = np.round([float(r['cpt_hits'])/360 - float(r['cpt_fa'])/360 for i,r in data.iterrows()],3)
# flip signs of variables so they are "in the same direction" in regards to underlying features of interest
data.loc[:,'scwt_conflict_acc_effect'] = data.loc[:,'scwt_conflict_acc_effect'] * -1
data.loc[:,'ant_conflict_acc_effect'] = data.loc[:,'ant_conflict_acc_effect'] * -1
data.loc[:,'sr_rec_explicitlearning_rt'] = data.loc[:,'sr_rec_explicitlearning_rt'] * -1
data.loc[:,'sr_enc_priming_rt'] = data.loc[:,'sr_enc_priming_rt'] * -1
data.loc[:,'cpt_fa'] = data.loc[:,'cpt_fa'] * -1

# do some useful renaming
data = data.rename(columns = rename_dict)

#separate into
survey_data = data.drop(data.columns[data.columns.str.contains('T.')], axis = 1)
task_data = data.drop(data.columns[data.columns.str.contains('S.')], axis = 1)


data = {'all_data': data, 'survey_data': survey_data, 'task_data': task_data, 'verbose_lookup': verbose_dict}
pickle.dump(data,open('../Data/subset_data.pkl', 'wb'), protocol = 2)