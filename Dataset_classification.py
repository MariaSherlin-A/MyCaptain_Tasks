import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


acs_data = pd.read_csv('acs_2019.csv')


acs_data.dropna(inplace=True)  # remove rows with missing values
acs_data['income'] = acs_data['income'].astype(float)  # convert income to float


def gini_coefficient(x):
    x = np.sort(x)
    n = len(x)
    gini = 0
    for i in range(n):
        gini += (2 * i - n + 1) * x[i]
    gini = gini / (n * np.sum(x))
    return gini

acs_data['gini'] = acs_data.groupby('state')['income'].transform(gini_coefficient)

plt.hist(acs_data['income'], bins=50)
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Distribution of Income')
plt.show()

sns.barplot(x='state', y='poverty_rate', data=acs_data)
plt.xlabel('State')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rates by State')
plt.show()

px.scatter(acs_data, x='state', y='gini', color='state')
px.update_layout(title='Income Inequality by State (Gini Coefficient)')
px.show()

px.scatter(acs_data, x='income', y='poverty_rate', color='state', hover_name='state')
px.update_layout(title='Income and Poverty by State')
px.show()