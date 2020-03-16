

### Importing library
import matplotlib.pyplot as plt
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataiku import insights

import sys
from itertools import chain
from dataiku import pandasutils as pdu
from dataiku.customrecipe import *

import json
import os
osp = os.path
import sys, json, os.path as osp, logging
from dataiku.doctor.utils import ProgressListener, unix_time_millis
from dataiku.doctor.prediction_entrypoints import *
import dataiku.doctor.utils as utils
from dataiku.doctor.preprocessing_handler import *
import dataiku.doctor.utils as utils
import plotly.graph_objects as go
import plotly.offline as py

import shap

## Loading inputs
input_A_names = get_input_names_for_role('train')
# The dataset objects themselves can then be created like this:
input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]
input_A_datasets=input_A_datasets[0]

# For outputs 1:
output_A_names = get_output_names_for_role('main_output')
output_A_datasets = [dataiku.Dataset(name) for name in output_A_names]

target1=get_recipe_config()['target1']
target2=get_recipe_config()['target2']


## Loading and importing dataiku ml model 
model_name = get_input_names_for_role('input_model')[0]
model = dataiku.Model(model_name)
my_predictor = model.get_predictor()
my_clf = my_predictor._clf


# Loading Training Data
data = input_A_datasets.get_dataframe()

### Loading model parameeter from URL
data_dir = os.environ['DIP_HOME']
data_dir
### Getting information from model

url1 = get_recipe_config()['url']
#url1="https://dss-amer.pfizer.com/projects/GBSUSVACCINEMODEL/analysis/RYrvXZCT/ml/p/XJ7pRh5L/A-GBSUSVACCINEMODEL-RYrvXZCT-XJ7pRh5L-s88-pp3-m1/report/#summary"
split = url1.split('projects/')[1].split('/')
project_key = split[0]
print(project_key)
analysis_id = split[2]
print(analysis_id)
modeling_id = split[5]
print(modeling_id)
print(split)
model_session = split[6].split('-')
#print(model_session)
session = model_session[-3]
print(session)
pp = model_session[-2]
print(pp)

## Getting access to pre-processed training data
exec_folder = osp.join(data_dir, 'analysis-data', project_key, analysis_id, modeling_id, 'sessions', session)
#exec_folder
core_params = json.load(open(osp.join(exec_folder, "core_params.json")))
#core_params
preprocessing_params = json.load(open(osp.join(exec_folder, pp, "rpreprocessing_params.json")))
preprocessing_params

preproc_handler = PredictionPreprocessingHandler.build(core_params, preprocessing_params, exec_folder)
collector = PredictionPreprocessingDataCollector(data, preprocessing_params)
collector_data = collector.build()
preproc_handler.collector_data = collector_data

pipeline = preproc_handler.build_preprocessing_pipeline(with_target=True)
transformed_full = pipeline.fit_and_process(data)
transformed_df = transformed_full['TRAIN'].as_dataframe()
transformed_df1=transformed_df
transformed_df['target'] = transformed_full['target']
transformed_df1=transformed_df1.drop(['target'], axis=1)

import timeit
start = timeit.default_timer()

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(my_clf)
shap_values = shap.TreeExplainer(my_clf).shap_values(transformed_df1)

directionality_visual=shap.summary_plot(shap_values, transformed_df1,plot_size="auto")

#insights.save_figure("my-matplotlib-explicit-plot", directionality_visual)


stop = timeit.default_timer()

print('Time: ', stop - start)  

## Shap directionality

df_shap=shap_values
df=transformed_df1 
#import matplotlib as plt and Make a copy of the input data
try:
        shap_v = pd.DataFrame(df_shap)
except:
        for var in df_shap:
            shap_v = pd.DataFrame(var)
            
feature_list = df.columns
shap_v.columns = feature_list
df_v = df.copy().reset_index().drop('index',axis=1)
    
# Determine the correlation in order to plot with different colors
corr_list = list()
for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
# Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
corr_df.columns  = ['Variable','Corr']
corr_df['Sign'] = np.where(corr_df['Corr']>=0,'green','blue')
    
# Plot it
shap_abs = np.abs(shap_v)
k=pd.DataFrame(shap_abs.mean()).reset_index()
k.columns = ['Variable','Featue_Importance']
k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
k2 = k2.sort_values(by='Featue_Importance',ascending = True)
colorlist = k2['Sign']

### negleting NA column attri
k3=pd.DataFrame([ x.split(':') for x in k2['Variable'].tolist() ])
    
k3['combined'] = k3[0].astype(str)+':'+k3[1].astype(str)+":"+k3[2].astype(str)
k3['Variable1'] = np.where(k3[0] !="dummy", k3[0],k3['combined'])
k2=k2.set_index('Variable')
k3=k3.set_index('Variable1')
    
result=pd.concat([k2,k3], axis=1, join='inner')
result1=result[result[2] != 'N/A']
result2= result1.drop(result1.columns[[3, 4, 5,6]], axis=1) 
result2['variable'] = result2.index


result2=result2.drop(['Corr'], axis=1)
result2['target_directionality']=np.where(result2['Sign']=='blue',target1,target2)



## writing dataset
output_A_datasets=output_A_datasets[0]
output_A_datasets.write_with_schema(result2)


