import math
from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

def info(df: pd.DataFrame, sub_df: pd.DataFrame, target_name: str) -> float:
    if df[target_name].nunique() == 0:
        return 0
    interval_count = 1 + int(math.log(df[target_name].nunique(), 2))
    left_border = df[target_name].min()
    right_border = df[target_name].max()
    step = (right_border - left_border) / interval_count

    vals_in_interval = [0] * interval_count
    value_counts = sub_df[target_name].value_counts()
    unq_vals = list(value_counts.index)
    if step != 0:
        for i in range(len(unq_vals)):
            interval_index = int((unq_vals[i] - left_border) // step)
            if interval_index == interval_count:
                interval_index -= 1
            vals_in_interval[interval_index] += value_counts[unq_vals[i]]

    s = 0.0
    for val in value_counts:
        tmp = val / sub_df[target_name].count()
        if tmp != 0:
            s += tmp * math.log(tmp, 2)
    return -s


def info_a(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    if df[attr_name].nunique() == 0:
        return 0
    interval_count = 1 + int(math.log(df[attr_name].nunique(), 2))
    left_border = df[attr_name].min()
    right_border = df[attr_name].max()
    step = (right_border - left_border) / interval_count
    left_borders = []
    for i in range(interval_count):
        left_borders.append(left_border + i * step)

    s = 0.0
    for border in left_borders:
        sub_df = df.loc[((df[attr_name] >= border) & (df[attr_name] < border + step)) |
                        (df[attr_name] == right_border)]
        s += info(df, sub_df, target_name)
    return s


def split_info(df: pd.DataFrame, attr_name: str) -> float:
    s = 0.0
    for val in df[attr_name].value_counts():
        tmp = val / df[attr_name].count()
        s += tmp * math.log(tmp, 2)
    return s


def gain(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    return info(df, df, target_name) - info_a(df, attr_name, target_name)


def gain_ratio(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    return gain(df, attr_name, target_name) / split_info(df, attr_name)

def data_set_gain_ratio(df: pd.DataFrame, target_name: str, num_target_columns: int) -> pd.Series:
    gain_ratio_list = []
    for col_name in df.columns[0:-num_target_columns]:
        gain_ratio_list.append(gain_ratio(df, col_name, target_name))
    return pd.Series(gain_ratio_list, index=df.columns[0:-num_target_columns])

PRINT_ALLOWED = True
MAX_MISSES_PERCENTS = 45
COLS = 1
ROWS = 0


def show_corr(df: DataFrame):
    width = 30
    height = 10
    sns.set(rc={'figure.figsize': (width, height)})
    sns.heatmap(df.corr(), annot=True, linewidths=3, cbar=False)
    plt.show()


def show_distributions(df: DataFrame, df_stat: DataFrame):
    for i in df.columns:
        plt.figure(i)
        sns.histplot(df[i], kde=True, stat="density")
        interquantile_range = df_stat.loc['Interquantile range'][i]
        plt.axvline(df_stat.loc['Quantile 1'][i] - 1.5 * interquantile_range, color="indigo", ls='--')
        plt.axvline(df_stat.loc['Quantile 1'][i], color="dodgerblue", ls='--')
        plt.axvline(df_stat.loc['Average'][i], color="red", ls='--')
        plt.axvline(df_stat.loc['Median'][i], color="goldenrod", ls='--')
        plt.axvline(df_stat.loc['Quantile 3'][i], color="dodgerblue", ls='--')
        plt.axvline(df_stat.loc['Quantile 3'][i] + 1.5 * interquantile_range, color="indigo", ls='--')
        plt.show()


def get_data_frame() -> DataFrame:
    df = pd.read_excel('ID_data_mass_18122012.xlsx', sheet_name='VU', skiprows=[0,2])
    df = df.drop(['Unnamed: 0', 'Unnamed: 1'], axis=COLS)
    return df


def get_frame_statistics(df: DataFrame) -> DataFrame:
    col_len = len(df.index)
    col_filled_len = df.count()
    filled_part = ((col_len - col_filled_len) / col_len) * 100
    minimum = df.min()
    q1 = df.quantile(q=0.25, )
    average = df.mean()
    median = df.median()
    q3 = df.quantile(q=0.75, )
    maximum = df.max()
    standard_deviation = df.std()
    unique_count = df.nunique()
    interquantile_range = q3 - q1
    frame = pd.concat([col_filled_len, filled_part, minimum, q1, average, median, q3, maximum, standard_deviation,
                       unique_count, interquantile_range], axis=1, join="inner")
    frame = frame.T
    f = pd.DataFrame(frame)

    f.index = ['Count', 'Unfilled percentage', 'Minimum', 'Quantile 1', 'Average', 'Median', 'Quantile 3',
               'Maximum', 'Standard deviation', 'Unique count', 'Interquantile range']
    return f


def remove_cols_with_many_misses(df: DataFrame, targets: list[str]) -> DataFrame:
    too_little_data_cols = []
    for col in df.columns:
        unfilled = df[col]['Unfilled percentage']
        if unfilled >= MAX_MISSES_PERCENTS and not targets.__contains__(col):
            too_little_data_cols.append(col)
    if PRINT_ALLOWED:
        print('Drop columns with many misses: ', too_little_data_cols)
    return df.drop(too_little_data_cols, axis=COLS)


def remove_cols_with_little_unique(df: DataFrame, targets: list[str]) -> DataFrame:
    too_little_unique_cols = []
    for col in df.columns:
        unique_count = df[col]['Unique count']
        if unique_count == 1 and not targets.__contains__(col):
            too_little_unique_cols.append(col)
    if PRINT_ALLOWED:
        print('Drop columns with little unique: ', too_little_unique_cols)
    return df.drop(too_little_unique_cols, axis=COLS)


def fill_blanks(df: DataFrame, df_stat: DataFrame, targets: List[str]) -> DataFrame:
    for col in df.columns:
        if col in targets:
            continue
        for i in df[col].keys():
            if df[col][i] is None or np.isnan(df[col][i]):
                df[col][i] = df_stat[col]['Median']
    return df

def combine_kgf(df: DataFrame) -> DataFrame:
    for row_num in df['КГФ.1'].keys():
        if not np.isnan(df['КГФ.1'][row_num]):
            df['КГФ'][row_num] = df['КГФ.1'][row_num] * 1000
    return df.drop('КГФ.1', axis=COLS)


def remove_empty_target(df: DataFrame, targets: List[str]) -> DataFrame:
    to_remove = []
    for i in df['КГФ'].keys():
        no_targets = True
        for col in targets:
            if not (df[col][i] is None or np.isnan(df[col][i])):
                no_targets = False
        if no_targets:
            to_remove.append(i)
    print("Drop ", len(to_remove), " rows: ", to_remove)
    for row in to_remove:
        df = df.drop(row, axis=ROWS)
    return df


def show_gain_ratio(df: DataFrame, df_stat: DataFrame, target: str, targets: list[str]):
    df_igr = data_set_gain_ratio(df, target, len(targets)).to_frame()
    df_igr.plot(kind = 'barh')
    plt.show()


def main():
    mpl.use('TkAgg')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = get_data_frame()
    df = combine_kgf(df)
    # print(df.describe())
    targets = ['G_total', 'КГФ']
    df_stat = get_frame_statistics(df)
    df_stat = remove_cols_with_many_misses(df_stat, targets)
    df_stat = remove_cols_with_little_unique(df_stat, targets)
    df = df.filter(items=df_stat)
    df = remove_empty_target(df, targets)
    df_stat = df_stat.filter(items=df)
    df = fill_blanks(df, df_stat, targets)
    # show_distributions(df, df_stat) # trash in Ro_c
    # df = df.drop(151, axis=ROWS)
    # show_corr(df)
    df_stat = get_frame_statistics(df)
    # show_corr(df)
    print(df_stat)
    # Рзаб and Рзаб1 have almost the same correlations
    # Also Руст and Руст1
    # By IGR defined that Руст.1 > Руст, Рзаб.1 > Рзаб, Дебит воды > Дебит воды.1
    df = df.drop(['Рзаб', 'Руст', 'Дебит воды.1'], axis=COLS)
    df_stat = df_stat.filter(items=df)

    df_g_total = df.drop(df[df['G_total'].isnull()].index)
    df_g_total_stat = get_frame_statistics(df_g_total)
    show_gain_ratio(df_g_total, df_g_total_stat, 'G_total', targets) # Рсб, Рсб.1
    df_kgf = df.drop(df[df['КГФ'].isnull()].index)
    df_kgf_stat = get_frame_statistics(df_kgf)
    show_gain_ratio(df_kgf, df_kgf_stat, 'КГФ', targets) # Рсб, Рсб.1
    # show_distributions(df, df_stat)
    df_stat = df_stat.filter(items=df)
    df.to_excel("result.xlsx")
    df_stat.to_excel("statistics.xlsx")
    # Рлин, Рсб .1
    df_1 = df.loc[[1, 2, 3, 24, 32]]
    df_2 = pd.merge(df, df_1, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    df_1.to_csv('testing_dataset.csv')
    df_2.to_csv('training_dataset.csv')


if __name__ == '__main__':
    main()
