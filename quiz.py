import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("dataset/fuel_data.csv")
df.drop('Unnamed: 0', 1, inplace = True)
df = df.rename(columns= {'fuel_type_code_pudl':'fuel_type',
                    'plant_name_ferc1':'plant_name',
                    'report_year':'year', 'utility_id_ferc1':'utility_id',
                    'fuel_qty_burned':'fuel_burned',
                    'fuel_mmbtu_per_unit': 'mmbtu_per_unit',
                    'fuel_cost_per_unit_burned':'cpu_burned',
                    'fuel_cost_per_unit_delivered':'cpu_delivered',
                    'fuel_cost_per_mmbtu': 'cpu_mmbtu' })

print(df.head())
