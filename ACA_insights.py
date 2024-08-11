# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:36:16 2024

@author: akshi
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina' # nicer rendering of plots in retina displays

import sys
print(sys.version)

import numpy as np
print('Numpy version:', np.__version__)

import pandas as pd
print('Pandas version:', pd.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
print('Matplotlib version:', mpl.__version__)

import plotly.graph_objs as go

import seaborn as sns
sns.set()
pal = sns.hls_palette(10, h=.5)
sns.set_palette(pal)

import re

#Source: https://www.kaggle.com/cdc/health-care-access-coverage
#The health insurance coverage data was compiled from the US Department of Health and Human Services and 
#US Census Bureau. It seems to relate to surveys, maybe the American Community Survey? 

pre_ACA = pd.read_csv('HC_access_1995-2010.csv')

#Remove % from string values
pre_ACA['Yes'] = pre_ACA.Yes.apply(lambda x: re.sub('%', '', x))
pre_ACA['No'] = pre_ACA.No.apply(lambda x: re.sub('%', '', x))
#Get geolocation
pre_ACA['geolocation'] = pre_ACA['Location 1'].apply(lambda x: x.split('\n')[1].split(','))
#Get latitude from geolocation
pre_ACA['lat'] = pre_ACA.geolocation.apply(lambda x: re.sub('\(', '', x[0]))
def get_long(x):
    return re.sub('\)|\s', '', x[1]) if len(x)>1 else ''
#Get longitude from geolocation
pre_ACA['long'] = pre_ACA.geolocation.apply(lambda x: get_long(x))
#Drop redundant columns
pre_ACA = pre_ACA.drop('geolocation', axis=1)
pre_ACA = pre_ACA.drop('Location 1', axis=1)
#Convert string datatypes to numeric
for col in ['Yes', 'No', 'lat', 'long']:
    pre_ACA[col] = pd.to_numeric(pre_ACA[col])
pre_ACA.tail()

len(sorted(pre_ACA.State.unique()))#50 states + DC + PR, Guam, VI + 2 Nationwide values

pre_ACA.Year.unique()
pre_ACA.Condition.unique()
pre_ACA.Category.unique() #redundant

#'Do you have any kind of health care coverage?' because it's not limited by the age group between 18 and 64.
pre_ACA[pre_ACA.Condition == 'Do you have any kind of health care coverage?'][['Yes', 'No']].describe()

#Print stats for comparison.
pre_ACA[pre_ACA.Condition != 'Do you have any kind of health care coverage?'][['Yes', 'No']].describe()


#Select a subset of the data
subset = pre_ACA[(pre_ACA.State.str.contains('Nationwide') == False)\
        & (pre_ACA.Condition == 'Do you have any kind of health care coverage?')] #844 rows
#Drop redundant columns
subset = subset.drop('Category', axis=1)
subset = subset.drop('Condition', axis=1)
#Load file with state codes
state_codes = pd.read_csv('state_codes.csv')
state_codes = state_codes[['code','state']]
state_codes.columns = ['code', 'State']
state_codes.tail()

#Merge state code into dataset
subset= subset.merge(state_codes, how='left', on='State')
#Check data
subset[23:38]

# Save the subset to a CSV file
subset.to_csv('subset_health_care_coverage.csv', index=False)

print("Subset saved to 'subset_health_care_coverage.csv'")


#Create pivot table with percentage of insured people per state per year
has_insurance = subset.pivot('code','Year','Yes')
has_insurance.head(10)

has_insurance = subset.pivot(index='code', columns='Year', values='Yes')

# Check the pivot table
print(has_insurance.head())



#Define lists with US Census regions

Northeast = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'Delaware', 
             'New Jersey', 'New York', 'Pennsylvania']

Midwest = ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 
           'Nebraska', 'North Dakota', 'South Dakota']

South = ['Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'District of Columbia',
         'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 
         'Oklahoma', 'Texas']

West = ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming', 'Alaska', 
        'California', 'Hawaii', 'Oregon', 'Washington']

Territories = ['Puerto Rico', 'Guam', 'Virgin Islands']

#Create series with grouped data per region
northeast = subset[subset.State.isin(Northeast)].groupby(['Year']).Yes.mean()
midwest = subset[subset.State.isin(Midwest)].groupby(['Year']).Yes.mean()
south = subset[subset.State.isin(South)].groupby(['Year']).Yes.mean()
west = subset[subset.State.isin(West)].groupby(['Year']).Yes.mean()
territories = subset[subset.State.isin(Territories)].groupby(['Year']).Yes.mean()

#Create plot with percentage of people who have answered to have some form of insurance (mean per US census region)

fig, ax = plt.subplots(1,1, figsize=(16, 6))

northeast.plot(label='Northeast', marker='o')
midwest.plot(label='Midwest', marker='D')
south.plot(label='South', marker='o')
west.plot(label='West', marker='D')
territories.plot(label='Territories', marker='s')

ax.set_facecolor('#F9F9F9')
ax.grid(color='#E4E4E4', linestyle='dotted', linewidth=1, axis ='y')

plt.ylim(75, 100)
plt.title('Percentage of People Insured Over the Years', fontsize=16, color='navy')
plt.xlabel('')
plt.tick_params(labelsize=12)
plt.legend(loc='upper right');




# Load second data
post_ACA = pd.read_csv('HC_access_afterACA.csv')

# Columns with percentage values
cols = ['Uninsured Rate (2010)', 'Uninsured Rate (2015)', 'Uninsured Rate Change (2010-2015)']

# Remove % from string values
for col in cols:
    post_ACA[col] = post_ACA[col].apply(lambda x: re.sub('%', '', x))

# Remove $ from string values
post_ACA['Average Monthly Tax Credit (2016)'] = post_ACA['Average Monthly Tax Credit (2016)'].apply(lambda x: x[1:-1])

# Convert string datatypes to numeric
for col in cols:
    post_ACA[col] = pd.to_numeric(post_ACA[col])
post_ACA['Average Monthly Tax Credit (2016)'] = pd.to_numeric(post_ACA['Average Monthly Tax Credit (2016)'])

# Remove trailing whitespace from state names
post_ACA.State = post_ACA.State.apply(lambda x: x.rstrip())
post_ACA.info()

# Check unique states
print(len(sorted(post_ACA.State.unique()))) # 50 states + DC + United States

# Load state codes
state_codes = pd.read_csv('state_codes.csv')
state_codes = state_codes[['code', 'state']]
state_codes.columns = ['code', 'State']

# Drop existing 'code' column if it exists
if 'code' in post_ACA.columns:
    post_ACA.drop(columns=['code'], inplace=True)

# Merge state code into dataset
post_ACA = post_ACA.merge(state_codes, how='left', on='State')

# Create column with the total insured population per state. Assign code for United States.
post_ACA.loc[post_ACA['State'] == 'United States', 'code'] = 'US'

# Calculate the total insured population
post_ACA['insured_pop'] = post_ACA.iloc[:, [5, 6, 11, 13]].sum(axis=1)

# Calculate the proportion of enrollment change per state between 2013 and 2016
# from the country's total new enrollments for the period
total_medicaid_change = post_ACA['Medicaid Enrollment Change (2013-2016)'][post_ACA['code'] == 'US'].values[0]
post_ACA['medicaid_delta'] = (post_ACA['Medicaid Enrollment Change (2013-2016)'] / total_medicaid_change) * 100

# Check data
print(post_ACA.tail())

# Save the subset to a CSV file
post_ACA.to_csv('After_ACA.csv', index=False)

# Sort by 'Uninsured Rate (2010)' and select top 19 rows
sorted_post_ACA = post_ACA.iloc[:, :4].sort_values(by='Uninsured Rate (2010)', ascending=False).head(19)
print(sorted_post_ACA)

# Sort by 'Uninsured Rate Change (2010-2015)' and then by 'Uninsured Rate (2010)'
uninsured_rate = post_ACA.iloc[:, [1, 2, 3, -3]]\
    .sort_values(by='Uninsured Rate Change (2010-2015)')\
    .head(17)\
    .sort_values(by='Uninsured Rate (2010)', ascending=False)

print(uninsured_rate)

#Create plot
ax = uninsured_rate.ix[:,:2].plot.bar(figsize = (15,7), fontsize=12, rot=0)
ax.set_facecolor('#F9F9F9')
ax.set_xticklabels(uninsured_rate.code)

plt.ylabel('Uninsured Population (%)', fontsize = 14)
plt.title('States with the Biggest Drops in Uninsured Population', fontsize=19, color='navy');

import matplotlib.pyplot as plt

# Plot the data
ax = uninsured_rate.iloc[:, :2].plot.bar(figsize=(15, 7), fontsize=12, rot=0)
ax.set_facecolor('#F9F9F9')
ax.set_xticklabels(uninsured_rate['code'])

plt.ylabel('Uninsured Population (%)', fontsize=14)
plt.title('States with the Biggest Drops in Uninsured Population', fontsize=19, color='navy')
plt.show()

#List dataset columns with insured population breakdown by source of insurance
cols = [u'code', u'Employer Health Insurance Coverage (2015)',
       u'Marketplace Health Insurance Coverage (2016)',
       u'Medicaid Enrollment (2016)', u'Medicare Enrollment (2016)',
       u'insured_pop']

# Select the necessary columns
insurance_type = post_ACA.loc[:, cols]

# Calculate total insured population and share of government coverage for each state
insurance_type['single_payer'] = insurance_type.loc[:, ['Medicaid Enrollment (2016)', 'Medicare Enrollment (2016)']].sum(axis=1) / insurance_type['insured_pop']
insurance_type['market'] = 1 - insurance_type['single_payer']

# Breakdown for each source of insurance
insurance_type['prop_employer'] = insurance_type['Employer Health Insurance Coverage (2015)'] / insurance_type['insured_pop']
insurance_type['prop_marketplace'] = insurance_type['Marketplace Health Insurance Coverage (2016)'] / insurance_type['insured_pop']
insurance_type['prop_medicaid'] = insurance_type['Medicaid Enrollment (2016)'] / insurance_type['insured_pop']
insurance_type['prop_medicare'] = insurance_type['Medicare Enrollment (2016)'] / insurance_type['insured_pop']

insurance_type.tail()

insurance_type.to_csv('insurance_type.csv', index=False)

#Print stats for US:
insurance_type.iloc[:-1].single_payer.describe() #mean = ~ 41%

#States where the government already covers more than 50% of the insured population
insurance_type[insurance_type.single_payer >= 0.50]

#State where the government represents the lowest propertion of insurance source
insurance_type[insurance_type.single_payer < 0.25]


#Create plot. National mean = 41% of the population covered by Medicaid or Medicare (26 states above mean).
ax = insurance_type[insurance_type.single_payer > .4].single_payer.sort_values(ascending=False).\
plot.bar(figsize=(16,6), fontsize=11, rot=0)

labels = insurance_type.sort_values(by='single_payer', ascending=False)['code']
ax.set_xticklabels(labels)
ax.set_facecolor('#F9F9F9')
ax.grid(color='#E4E4E4', linestyle='dotted', linewidth=1, axis ='y')

plt.text(21, .52, '32 states (Medicaid & Medicare)', color='orange', size= 16, weight ='bold')
plt.ylim(.2,.6)
plt.title('States where > 40% of the Population is Insured by the Government (2016)', fontsize=19, color='navy');



import matplotlib.pyplot as plt

# Filter and sort the data
filtered_data = insurance_type[insurance_type.single_payer > 0.4].sort_values(by='single_payer', ascending=False)

# Plot the data
ax = filtered_data.single_payer.plot.bar(figsize=(16, 6), fontsize=11, rot=0)

# Set the x-tick labels
labels = filtered_data['code']
ax.set_xticklabels(labels)

# Customize the plot
ax.set_facecolor('#F9F9F9')
ax.grid(color='#E4E4E4', linestyle='dotted', linewidth=1, axis='y')

plt.text(21, 0.52, '32 states (Medicaid & Medicare)', color='orange', size=16, weight='bold')
plt.ylim(0.2, 0.6)
plt.title('States where > 40% of the Population is Insured by the Government (2016)', fontsize=19, color='navy')
plt.show()


#Scale size of insured population to be used as bubble dimension

#color= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
size = insurance_type.iloc[:-1, :].insured_pop/500000. #disregard last row which is data for the US
size.min(), size.max()
(1.039574, 77.280572000000006)
#Bubble chart: proportion of single payer per state (similar to Fig3a but showing all states). Nationwide mean: 41%
#The bubble size is proportional to the total insured population in each state.

trace = go.Scatter (x = insurance_type.iloc[:-1, :].code, 
                    y = insurance_type.iloc[:-1, :].single_payer,
                    text = 'Proportion of Medicare + Medicaid',
                    marker = dict(size = size, 
                                  color=range(len(insurance_type)-1), 
                                  opacity= 0.7, 
                                  colorscale='Viridis'), 
                    mode = 'markers')


layout = dict(
    title = 'Proportion of Insured Population Covered by the Government<br>(bubble size is proportional to population)',
    xaxis=dict(tickangle=-90)
)


data=[trace]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='proportion-single-payer')


#Create list with (year, % uninsured population) tuples for every 5 years
uninsured = []
years = [1995, 2000, 2005]

for y in years:
    uninsured.append((y, subset[subset.Year==y].No.mean()))
    
uninsured.append((2010, post_ACA['Uninsured Rate (2010)'][51]))
uninsured.append((2015, post_ACA['Uninsured Rate (2015)'][51]))

dict(uninsured)

#Create summary dataframe containing data about population and GDP for each year.
#Source of population estimates: US Census Bureau via http://www.multpl.com/united-states-population/table

gdp_per_capita = [38.01, 44.63, 47.96, 47.86, 51.22] #chained to 2009 dollars (inflation adjusted)
pop_millions = [266.28, 282.16, 295.52, 308.11, 319.7]

summary_df = pd.DataFrame(uninsured, columns = ['year', 'uninsured_%'])
summary_df['gdp_thousands_$'] = pd.Series(gdp_per_capita)
summary_df['pop_millions'] = pd.Series(pop_millions)
summary_df['pop_insured'] = ((100 - summary_df['uninsured_%'])/100.0)* summary_df.pop_millions

summary_df

import matplotlib.pyplot as plt

# Define function to automatically add annotated labels on bar plots
def annotate_labels(ax, labels_list, **kwargs):
    # Get y-axis height to calculate label position from
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom
    
    rects = ax.patches

    for rect, label in zip(rects, labels_list):
        height = rect.get_height()
        p_height = (height / y_height)  # Fraction of axis height taken up by this rectangle
        label_position = height + (y_height * 0.01)        
        ax.text(rect.get_x() + rect.get_width() / 2., label_position, label, kwargs)
    return None

# Define arguments for annotate_labels function to be used to create the plot with data before and after the ACA
labels_list = [str(x) + '%' for x in summary_df['uninsured_%'].values.round(1)]
kwargs = {'fontsize': 13, 'ha': 'center', 'va': 'bottom', 'weight': 'bold'}

# Create comparison plot (before and after ACA (2015))
fig, ax = plt.subplots(figsize=(16, 6))

# Bars representing total US population at each year
ax.bar(summary_df.year, summary_df.pop_millions, color='#00E5EE', width=3, label='Uninsured Proportion from Total Population')
ax.set_facecolor('#f6f6f6')
ax.tick_params(labelsize=14)
ax.legend(loc='upper left', fontsize=12)
plt.ylim(0, 350)
plt.ylabel('Population (Millions)', fontsize=13)

# Superimposing bars with total insured population in the US
ax2 = ax.twinx()
ax2.bar(summary_df.year, summary_df.pop_insured, color='#003366', width=2.9)
ax2.get_yaxis().set_ticks([])
annotate_labels(ax2, labels_list, **kwargs)
plt.ylim(0, 350)

plt.show()

summary_df.to_csv('summary_df.csv', index=False)