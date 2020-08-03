import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

style.use('ggplot')

#reads the dataset into the script
pdf = pd.read_csv("dataset/fuel_data.csv")
pdf.drop('Unnamed: 0', 1, inplace = True)

#because the names were too long, i reduced their lengths
pdf = pdf.rename(columns= {'fuel_type_code_pudl':'fuel_type',
                    'plant_name_ferc1':'plant_name',
                    'report_year':'year', 'utility_id_ferc1':'utility_id',
                    'fuel_qty_burned':'fuel_burned',
                    'fuel_mmbtu_per_unit': 'mmbtu_per_unit',
                    'fuel_cost_per_unit_burned':'cpu_burned',
                    'fuel_cost_per_unit_delivered':'cpu_delivered',
                    'fuel_cost_per_mmbtu': 'cpu_mmbtu' })

print(pdf.head())

"""
Performing an Exploratory Data Analysis (EDA) on the dataset
To check for the year in which the cost per unit burned of gas was at its peak
there are several ways to do this
Option A:
    We could iterate through the years, and find the sum for each filtered
    dataframe consisting of the desired fuel type keeping the cost per unit
    burned feature constant
Option B:
    We can also use the pandas function, groupby
"""
#Oprion A
highest, list_of_sum = 0, []
for year in pdf['year'].unique():
    total = pdf['cpu_burned'][(pdf.year== year)& (pdf.fuel_type=='gas')].sum()
    if total>highest:
        highest = total
        highestyear = year
    list_of_sum.append(total)

print(f'The year when gas cost per unit was at its highest was {highestyear}'+
        f'at {highest} mcf')

#Option B
df = pdf[pdf['fuel_type']=='gas'].groupby('year').sum()
list_of_sum_2 = list(df['cpu_burned'].round(decimals =3))

print(max(list_of_sum_2))
#You can see the results are equal


#To plot this data, you can do the following

sumlist = np.array(list_of_sum)           #this converts the list into a numpy array
plt.figure(figsize = (10,5))
plt.plot(pdf.year.unique(), sumlist)
plt.xlabel('Yyears')
plt.ylabel('Volume in mcf')
plt.title('Plot of the distribution of Cost per unit gas burned per year')
plt.show()
