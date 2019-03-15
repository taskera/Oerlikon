# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:32:48 2019

@author: acostalago
"""
import matplotlib.pyplot as plt
import pandas as pd
from PreprocessData import dataset, y
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm


# Constant
GREEN = '#00441b'

""" Definition of functions"""

def plot_bar_df(df, name, colormap='PiYG_r'):
    df=pd.DataFrame(df)
    df = df.unstack()
    df = df.div(df.sum(axis = 1), axis = 0)
    ax = df.plot(kind='bar', stacked=True, colormap=colormap, edgecolor = "black", fontsize=16)
    ax.legend(['Functional', 'Non functional'])
    ax.set_xlabel(name, fontsize=16)
    ax.set_ylabel('Percentage', fontsize=16)
    ax.axhline(y=0.5, color='k', linestyle='--', lw=2)
    
# Chi-Squared Test of Independence
    # Given two categorical variables it determines if they are significantly related to each other
    
def Chi_Square(df, name):
    chi = 0
    df=pd.DataFrame(df)
    df = df.unstack(level=0)
    df['row_total'] = df.sum(axis = 1)
    df2= df.sum(axis = 0)
    df2.name = 'Column_total'
    df = df.append(df2)
    f_obs = np.array([df.iloc[0][0:-1].values, df.iloc[1][0:-1].values])
    chi, p_value, n_var = stats.chi2_contingency(f_obs)[0:3]
    if p_value < 0.05:
        print("There is a statistical relationship between the functionality of the water supply and", name, ". P value: ", p_value)
    else:
        print("There is no statistical relationship between the functionality of the water supply and", name, ". P value: ", p_value)            
    return p_value

""" Data exploration """

dataset['status_group'].value_counts()
dataset['status_group'].value_counts().div(dataset['status_group'].count())
print("There is a 60% vs a 40% of functional vs non functional data. The dateset is not imbalance for modelling.")

# Preliminary visualization of the data
a = dataset.groupby('status_group').mean()
dataset.groupby(['status_group','installer']).size()
dataset.groupby(['status_group','permit']).size()
dataset.groupby(['status_group','payment_type']).size()
dataset.groupby(['status_group','quality_group']).size()
dataset.groupby(['status_group','quantity_group']).size()
dataset.groupby(['status_group','source_type']).size()
dataset.groupby(['status_group','source_class']).size()
dataset.groupby(['status_group','waterpoint_type_group']).size()
dataset.groupby(['status_group','basin']).size()
dataset.groupby(['status_group','public_meeting']).size()
dataset.groupby(['status_group','Year']).size()
dataset.groupby(['status_group','Type_class']).size()


""" Columns of interest: amount_tsh, installer, basin, population, 
public_meeting, scheme_management, permit, Year, Type_class , 
management_group, payment_type, quality_group, quantity_group, source_type 
source_class, waterpoint_type_group """

""" Plot boxplots"""
# Variable definition
x = ['Functional', 'Non functional']

# Status group
plt.bar(x, dataset['status_group'].value_counts(), color = GREEN)
plt.title('Functionality', fontsize=18)
plt.xlabel('Status', fontsize=16)
plt.ylabel('Count of functionality status', fontsize=16)
plt.show()
print("The dependent variable seems balanced")


# amount_tsh
plt.bar(x, a['amount_tsh'], color = GREEN)
plt.title('Functionality vs Total static head', fontsize=18)
plt.xlabel('Total static head', fontsize=16)
plt.ylabel('Mean of Static head', fontsize=16)
plt.show()
print("The average of total static head is higher in functional water point status")
logit_model=sm.Logit(y, dataset['amount_tsh'].values)
result=logit_model.fit()
print(result.summary2())


# installer
df = dataset.groupby(['installer','status_group']).size()
# Too many variables to plot plot_bar_df(df)
p_value = Chi_Square(df, 'installer')

# basin
df = dataset.groupby(['basin','status_group']).size()
plot_bar_df(df, 'basin')
print("There are two basin that are more likely to be broken")
p_value = Chi_Square(df, 'basin')

# population
plt.bar(x, a['population'], color = GREEN)
plt.title('Functionality vs population', fontsize=18)
plt.xlabel('population', fontsize=16)
plt.ylabel('Mean of population', fontsize=16)
plt.show()
print("It does not seem to be a difference depending on population")
logit_model=sm.Logit(y, dataset['population'].values)
result=logit_model.fit()
print(result.summary2())

# public_meeting
df = dataset.groupby(['public_meeting','status_group']).size()
plot_bar_df(df, 'public_meeting')
print("Those on public meetings are more likely to function")
p_value = Chi_Square(df, 'public_meeting')

# scheme_management
df = dataset.groupby(['scheme_management','status_group']).size()
plot_bar_df(df, 'scheme_management')
print("Those managed by SWC are more likely to fail")
p_value = Chi_Square(df, 'scheme_management')

# permit
df = dataset.groupby(['permit','status_group']).size()
plot_bar_df(df, 'permit')
print("Having a permit or not does not affect the functionality")
p_value = Chi_Square(df, 'permit')


# Year
df = dataset.groupby(['Year','status_group']).size()
plot_bar_df(df, 'Year', colormap = 'Greens_r')
print("Newer installations are more likely to function")
p_value = Chi_Square(df, 'Year')

# Type_class
df = dataset.groupby(['Type_class','status_group']).size()
plot_bar_df(df, 'Type_class')
print("Those working with gravity, hand pump or rope pump are more likely to function")
p_value = Chi_Square(df, 'Type_class')

# management_group
df = dataset.groupby(['management_group','status_group']).size()
plot_bar_df(df, 'management_group')
print("Those with unknown managedment groups are more likely to fail")
p_value = Chi_Square(df, 'management_group')

# payment_type
df = dataset.groupby(['payment_type','status_group']).size()
plot_bar_df(df, 'payment_type')
print("Those with no payment methods or unknown payment methods are more likely to fail")
p_value = Chi_Square(df, 'payment_type')

# quality_group
df = dataset.groupby(['quality_group','status_group']).size()
plot_bar_df(df, 'quality_group')
print("Those with salty or unknown quality are more likely to fail")
p_value = Chi_Square(df, 'quality_group')


# quantity_group
df = dataset.groupby(['quantity_group','status_group']).size()
plot_bar_df(df, 'quantity_group')
print("Those from dry or unknown sources are more likely to fail")
p_value = Chi_Square(df, 'quantity_group')

# source_type
df = dataset.groupby(['source_type','status_group']).size()
plot_bar_df(df, 'source_type')
print("Those from dams are more likely to fail")
p_value = Chi_Square(df, 'source_type')

# source_class
df = dataset.groupby(['source_class','status_group']).size()
plot_bar_df(df, 'source_class')
print("The source of the water is no indication of the functionality")
p_value = Chi_Square(df, 'source_class')

# waterpoint_type_group
df = dataset.groupby(['waterpoint_type_group','status_group']).size()
plot_bar_df(df, 'waterpoint_type_group')
print("Those from undefined waterpoints are more likely to fail")
print("This also shows that water from dams are more likely to work")
p_value = Chi_Square(df, 'waterpoint_type_group')


