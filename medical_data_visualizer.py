import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
## Calculate BMI
df['BMI']= df['weight']/((df['height']/100)**2)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)
## Drop BMI column
df = df.drop(columns=['BMI'])

# 3
## Normalize cholesterol and gluc values
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    ## Create DataFrame for cat plot using `pd.melt`
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    ## Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7
    ## Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=5, aspect=1)

    ## Set the axis labels and title
    g.set_axis_labels("variable", "total")
    g.set_titles("Cardio = {col_name}")
    g.fig.suptitle('Categorical Plot of Medical Examination Data', y=1.02)

    # 8
    ## Call the function to generate the plot
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    ## Clean the data
    df_heat =  df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    ## Calculate the correlation matrix
    corr = df_heat.corr()

    # 13
    ## Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    ## Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    ## Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', linewidths=0.5, ax=ax)
    
    # 16
    fig.savefig('heatmap.png')
    return fig
