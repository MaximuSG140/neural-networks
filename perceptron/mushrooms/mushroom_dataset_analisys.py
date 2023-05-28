import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('agaricus-lepiota.data', header=None)

def assign_numbers(df):
    mapping = {}
    for col in df.columns: 
        unique_letters = df[col].unique()
        mapping[col] = {letter: num + 1 for num, letter in enumerate(unique_letters)}
    return df.replace(mapping)

df = assign_numbers(df)
df = df.drop(df.columns[16], axis=1)
df = (df - df.min()) / (df.max() - df.min())

C = len(df.columns)
L = len(df.index)
CN = df.count() #количество
NP = ((L - CN) / L) * 100 #процент пропущенных значений
MN = df.min(numeric_only=True) #минимум
Q1 = df.quantile(q=0.25, numeric_only=True) #первый квартиль
MA = df.mean(numeric_only=True) #среднее значение
ME = df.median(numeric_only=True) #медиана
Q3 = df.quantile(q=0.75, numeric_only=True) #третий квартиль
MX = df.max(numeric_only=True) #максимум
ST = df.std(numeric_only=True) #стандартное отклонение
P = df.nunique() #мощность
IQ = Q3 - Q1 #интерквартильный размах

frame = pd.concat([CN, NP, MN, Q1, MA, ME, Q3, MX, ST, P, IQ], axis=1, join="inner")
frame = frame.T
f = pd.DataFrame(frame)
f.index=['Количество', 'Процент пропусков', 'Минимум', 'Первый квартиль','Среднее', 'Медиана', 'Третий квартиль', 'Максимум','Стандартное отклонение', 'Мощность', 'Интерквартильный размах']


for i in df.columns:
    plt.figure(i)
    sns.histplot(df[i],kde=True,stat="density")
    plt.axvline(f.iloc[3][i]-1.5*   f.iloc[10][i], color="indigo", ls='--') #q1-1.5*iqr
    plt.axvline(f.iloc[3][i], color="dodgerblue", ls='--') #первый квартиль
    plt.axvline(f.iloc[4][i], color="red", ls='--') #среднее
    plt.axvline(f.iloc[5][i], color="goldenrod", ls='--') #медиана
    plt.axvline(f.iloc[6][i], color="dodgerblue", ls='--') #третий квартиль
    plt.axvline(f.iloc[6][i]+1.5*f.iloc[10][i], color="indigo", ls='--') #q3+1.5*iqr
    plt.show()

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df.corr(), annot=True, linewidths=3, cbar=False)
plt.show()

training_dataset_size = round(len(df.index) * 0.7)
df1 = df.iloc[:training_dataset_size,:]
df2 = df.iloc[training_dataset_size:,:]
df1.to_csv('training_mushrooms_dataset.csv', index=False)
df2.to_csv('testing_mushrooms_dataset.csv', index=False)