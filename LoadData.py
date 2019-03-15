# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:54:10 2019

@author: acostalago
"""

import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('pump_data_1.csv')
dataset2 = pd.read_csv('pump_data_2.csv')

# Need to join the two datasets

dataset2['id'] = dataset2['id'].str.extract('(\d+)').astype(int)
dataset = pd.merge(dataset1,dataset2, on='id')

# Join similarly named variables

dataset.installer[dataset['installer'] == 'ACTIVE TANK CO LTD'] = 'ACTIVE TANK CO'
dataset.installer[dataset['installer'] == 'ACT'] = 'ACT MARA'
dataset.installer[dataset['installer'] == 'ADRA /Government'] = 'ADRA'
dataset.installer[dataset['installer'] == 'ADRA/Government'] = 'ADRA'
dataset.installer[dataset['installer'] == 'local technician'] = 'local'
dataset.installer[dataset['installer'] == 'local technitian'] = 'local'
dataset.installer[dataset['installer'] == 'local fundi'] = 'local'
dataset.installer[dataset['installer'] == '0'] = 'unknown'
dataset.installer[dataset['installer'] == '-'] = 'unknown'
dataset.installer[dataset['installer'] == 'not known'] = 'unknown'
dataset.installer[dataset['installer'] == 'sengerema water Department'] = 'sengerema Water Department'
dataset.installer[dataset['installer'] == 'plan int'] = 'plan Int'
dataset.installer[dataset['installer'] == 'ADP'] = 'ADP Busangi'
dataset.installer[dataset['installer'] == 'AMP contractor'] = 'AMP Contract'

# Separate year and type class
dataset['Year'] = dataset['constr_year_extract_type_class'].str.extract('(\d+)').astype(int)
dataset['Type_class'] = dataset['constr_year_extract_type_class'].str.replace('\d+', '').str.strip('_')

# Eliminate outliers
dataset = dataset[dataset.amount_tsh <= 200000]

# Eliminate nan values
dataset.installer[dataset['installer'].isnull()] = 'unknown'
dataset.public_meeting[dataset['public_meeting'].isnull()] = False
dataset.scheme_management[dataset['scheme_management'].isnull()] = 'unknown'
dataset.population[dataset['population'].isnull()] = 0
dataset.permit[dataset['permit'].isnull()] = False

# Fill the rest of NaN values with 0
dataset = dataset.fillna(0)

