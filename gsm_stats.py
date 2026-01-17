import pandas as pd
df = pd.read_csv('split_feature_dataset/dataset_all.csv')
print('GSM Statistics:')
print(f'  Min: {df["gsm"].min():.2f}')
print(f'  Max: {df["gsm"].max():.2f}')
print(f'  Mean: {df["gsm"].mean():.2f}')
print(f'  Std: {df["gsm"].std():.2f}')
