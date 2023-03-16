# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:33:28 2018

@author: atdadaev
"""

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Load the dataset into a DataFrame
df = pd.read_csv('C:/*/Education/MachineLearning/University of California Irvine/POKEMON/POKEMON.csv')

# Show the first five rows
print(df.head())

# Rename the # column to id, and convert all column labels to lower case
df.rename(columns={'#': 'id'}, inplace=True)
df.columns = df.columns.str.lower()

# Take a look at any duplicated rows via the id column
df[df.duplicated('id', keep=False)].head()

# Remove any Pokemon with duplicated id numbers except for the first instance
df.drop_duplicates('id', keep='first', inplace=True)

# Replace any missing values in type 2 with 'None'
df['type 2'].fillna(value='None', inplace=True)

# Select only the identity-related columns from original dataset and store them
# in a new table called pokedex
pokedex = df[['id', 'name', 'type 1', 'type 2', 'generation', 'legendary']]

# Join the original dataset with the pokedex table to generate a new table
# called statistics
statistics = pd.merge(
    df,
    pokedex,
    on='id'
).loc[:, ['id', 'hp', 'attack', 'defense', 'sp. atk', 'sp. def', 'speed',
          'total']]

print(pokedex.head())
print(statistics.head())

# Total number of Pokemon in each generation
# Improve readability
sns.set_context('notebook', font_scale=1.4)

# Display a categorial histogram and label axes
sns.factorplot(
    x='generation', 
    data=pokedex, 
    kind='count'
).set_axis_labels('Generation', '# of Pokemon');

# Display a categorical histogram faceted by primary type and label axes
sns.factorplot(
    x='generation',
    data=pokedex,
    col='type 1',
    kind='count',
    col_wrap=3
).set_axis_labels('Generation', '# of Pokemon');

plt.show()

# Identify the primary types that are also secondary types, and vice versa
unique_type1 = np.setdiff1d(pokedex['type 1'], pokedex['type 2'])
unique_type2 = np.setdiff1d(pokedex['type 2'], pokedex['type 1'])

# Display any unique primary types
print('Unique Type 1: ', end='')
if unique_type1.size == 0:
    print('No unique types')
else:
    for u in unique_type1:
        print(u)

# Display any unique secondary types
print('Unique Type 2: ', end='')
if unique_type2.size == 0:
    print('No unique types')
else:
    for u in unique_type2:
        print(u)

# Group by primary or secondary type, and compute the total number in each 
# group
type1, type2 = pokedex.groupby('type 1'), pokedex.groupby('type 2')
print('Type 1 count: {}'.format(len(type1)))
print('Type 2 count: {}'.format(len(type2)))

# Display a categorical histogram of primary types and sort by largest counts
sns.factorplot(
    y='type 1',
    data=pokedex,
    kind='count',
    order=pokedex['type 1'].value_counts().index,
    aspect=1.5,
    size=5,
    color='green'
).set_axis_labels('# of Pokemon', 'Type 1')

# Display a categorical histogram of secondary types and sort by largest counts
sns.factorplot(
    y='type 2',
    data=pokedex,
    kind='count',
    order=pokedex['type 2'].value_counts().index,
    aspect=1.5,
    size=5,
    color='purple'
).set_axis_labels('# of Pokemon', 'Type 2');

plt.show()

# Improve readability
sns.set_context('talk', font_scale=1.0)

# Exclude Pokemon species without a secondary type to better highlight results
dual_types = pokedex[pokedex['type 2'] != 'None']

# Display a heatmap of the different combinations of primary and secondary 
# types, and show the counts for each combination
sns.heatmap(
    dual_types.groupby(['type 1', 'type 2']).size().unstack(),
    linewidths=1,
    annot=True
);

plt.show()

# Improve readability
sns.set_context('notebook', font_scale=1.4)

# Select any Pokemon with only a primary type
single_types = pokedex[pokedex['type 2'] == 'None']

# Display a categorical histogram and sort by largest counts
sns.factorplot(
    y='type 1', 
    data=single_types,
    kind='count',
    order=single_types['type 1'].value_counts().index,
    aspect=1.5,
    size=5,
    color='grey'
).set_axis_labels('# of Pokemon', 'Type 1');

plt.show()

# Join the statistics table with the pokedex table based on id number to report
# the 10 highest total values
pd.merge(
    pokedex, 
    statistics, 
    on='id'
).sort_values('total', ascending=False).head(10)

# Remove the total column, move the id column to the row labels, standardize
# each statistic by converting values into a z-score, and store the results in
# a new table
std_stats = statistics.drop('total', axis='columns').set_index('id').apply(
    lambda x: (x - x.mean()) / x.std())

# Compute the strength as the sum of the z-scores for each statistic and store
# the result in a new column
std_stats['strength'] = std_stats.sum(axis='columns')

# Re-insert the id column back into the table
std_stats.reset_index(inplace=True)

# Join the std_stats table with the pokedex table based on id number to report
# the Pokemon with the 10 highest strength values
pd.merge(
    pokedex, 
    std_stats, 
    on='id'
).sort_values('strength', ascending=False).head(10)

# Join the std_stats table with the pokedex table based on id number to report
# the Pokemon with the 10 lowest strength values
pd.merge(
    pokedex,
    std_stats,
    on='id'
).sort_values('strength').head(10)

# Join the std_stats table with the pokedex table based on id number to report
# the non-legendary Pokemon with the 10 highest strength values
pd.merge(
    pokedex[~pokedex['legendary']],
    std_stats,
    on='id'
).sort_values('strength', ascending=False).head(10)

# Improve readability
sns.set_context('talk', font_scale=1.0)

# Join the std_stats table with the pokedex table based on id number
joined = pd.merge(
    pokedex,
    std_stats,
    on='id'
)

# Calculate the median strength of each combination of primary and secondary
# type
medians = joined.groupby(['type 1', 'type 2']).median().loc[:, 'strength']

# Display a heatmap of the median strength of each combination of primary and
# secondary type
sns.heatmap(
    medians.unstack(),
    linewidths=1,
    cmap='RdYlBu_r'
);

# Rearrange the medians table and sort by the strongest combination of primary
# and secondary types
medians.reset_index().sort_values('strength', ascending=False).head()

# Join the std_stats table with the pokedex table based on id number for
# non-legendary Pokemon only
joined_nolegs = pd.merge(
    pokedex[~pokedex['legendary']],
    std_stats,
    on='id'
)

# Calculate the median strength of each combination of primary and secondary
# type
medians = joined_nolegs.groupby(['type 1', 
                                 'type 2']).median().loc[:,'strength']

# Display a heatmap of the median strength of each combination of primary and
# secondary type for non-legendary Pokemon
sns.heatmap(
    medians.unstack(),
    linewidths=1,
    cmap='RdYlBu_r'
);
        
plt.show()
        
# Rearrange the medians table and sort by the strongest combination of primary
# and secondary types for non-legendary Pokemon
medians.reset_index().sort_values('strength', ascending=False).head()

# Display a heatmap of the distribution of statistics across primary types
sns.heatmap(
    joined.groupby('type 1').median().loc[:, 'hp':'speed'], 
    linewidths=1,
    cmap='RdYlBu_r'
);

plt.show()

# Display a heatmap of the distribution of statistics across primary types
# for non-legendary Pokemon
sns.heatmap(
    joined_nolegs.groupby('type 1').median().loc[:, 'hp':'speed'], 
    linewidths=1,
    cmap='RdYlBu_r'
);

plt.show()

def show_corr(x, y, **kwargs):
    # Calculate Pearson's coefficient for each pair of statistics
    (r, _) = stats.pearsonr(x, y)
    
    # Annotate Pearson's coefficient, rounded to two decimal places, on each 
    # pairwise plot
    ax = plt.gca()
    ax.annotate(
        'r = {:.2f}'.format(r),
        xy=(0.45, 0.85),
        xycoords=ax.transAxes
    )

# Improve readability
sns.set_context('paper', font_scale=1.5)

# Show scatterplots with linear regression and Pearson's coefficient for each 
# pair of statistics
sns.pairplot(
    data=joined_nolegs.loc[:, 'hp':'speed'],
    kind='reg'
).map_offdiag(show_corr);  
        
plt.show()        
        
# Identify the HP outliers     
joined_nolegs.loc[joined_nolegs['hp'] > 6, 'name']        
# Outliers with high defense        
joined_nolegs.loc[joined_nolegs['defense'] > 5, 'name']        
# Outliers with high special defense        
joined_nolegs.loc[joined_nolegs['sp. def'] > 5, 'name']        
        
# Strongest type of Pokemon: Ghost/Dragon        
stp = pokedex[(pokedex['type 1'] == 'Ghost') & (pokedex['type 2'] == 'Dragon')]     
# Strongest type of non-legendary Pokemon: Dragon/Flying
stnlp = pokedex[(pokedex['type 1'] == 'Dragon') & (pokedex['type 2'] == 'Flying')]
####################
print(stp)
print(stnlp)










