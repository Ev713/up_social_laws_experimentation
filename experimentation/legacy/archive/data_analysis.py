import json
import pandas
import re

from tabulate import tabulate

def read_data():
    exp_df = pandas.read_csv(
        '/home/evgeny/SocialLaws/up-social-laws/experimentation/logs/experiment_log_Oct-30-2024.csv')
    blocksworld_df = exp_df[exp_df.apply(lambda x: 'blocksworld' in x['name'], axis=1)]
    print('\nBLOCKSWORLD\n')
    print(blocksworld_df)
    zenotravel_sl_df = exp_df[exp_df.apply(lambda x: 'zenotravel' in x['name'] and 'sl' in x['name'], axis=1)]
    print('\nZENOTRAVEL_SL\n')
    print(zenotravel_sl_df)
    zenotravel_df = exp_df[exp_df.apply(lambda x: 'zenotravel' in x['name'] and not 'sl' in x['name'], axis=1)]
    print('\nZENOTRAVEL\n')
    print(zenotravel_df)
    driverlog_df = exp_df[exp_df.apply(lambda x: 'driverlog' in x['name'], axis=1)]
    print('\nDRIVERLOG\n')
    print(driverlog_df)
    grid_df = exp_df[exp_df.apply(lambda x: 'grid' in x['name'] and not 'sl' in x['name'], axis=1)]
    print('\nGRID\n')
    print(grid_df)
    grid_sl_df = exp_df[exp_df.apply(lambda x: 'grid' in x['name'] and 'sl' in x['name'], axis=1)]
    print('\nGRID_SL\n')
    print(grid_sl_df)


def add_new_old_cols(df):
    df = df.sort_values(by='name').reset_index(drop=True).drop('has_social_law', axis=1)
    slrc_old = df[df['slrc_is_old'] == True]
    slrc_old = slrc_old.rename(columns={'time': 'old_compilation'}, ).drop('slrc_is_old', axis=1)
    slrc_new = df[df['slrc_is_old'] == False]
    slrc_new = slrc_new.rename(columns={'time': 'new_compilation'}, ).drop('slrc_is_old', axis=1)
    df = pandas.merge(slrc_old, slrc_new, on='name', how='inner')
    df = df[['name', 'old_compilation', 'new_compilation']]
    return df


def add_sl_no_sl_columns(df, name_tag):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Filter rows where `name_tag` is in the 'name' column
    df = df[df['name'].str.contains(name_tag, na=False)]

    # Sort and reset the index
    df = df.sort_values(by='name').reset_index(drop=True)

    # Split into 'sl' and 'no sl' DataFrames based on 'sl' in the name
    df_sl = df[df['name'].str.contains('sl', na=False)].copy()
    if len(df_sl) == 0:
        return df
    df_no_sl = df[~df['name'].str.contains('sl', na=False)].copy()

    # Remove 'sl' from names in `df_sl`
    df_sl['name'] = df_sl['name'].str.replace('sl_', '', regex=False)

    # Rename columns in `df_sl` to match the target names
    df_sl = df_sl.rename(columns={
        'old_compilation': 'old_compilation_with_sl',
        'new_compilation': 'new_compilation_with_sl'
    })

    # Select only the required columns in `df_sl`
    df_sl = df_sl[['name', 'old_compilation_with_sl', 'new_compilation_with_sl']]

    # Merge `df_sl` with `df_no_sl` on 'name'
    df_merged = pandas.merge(df_sl, df_no_sl, on='name', how='inner')

    return df_merged


def transform_data(df):
    df = df.sort_values(by='name').reset_index(drop=True).drop('has_social_law', axis=1)
    slrc_old = df[df['slrc_is_old'] == True]
    slrc_old = slrc_old.rename(columns={'time': 'old_compilation'}, ).drop('slrc_is_old', axis=1)
    slrc_new = df[df['slrc_is_old'] == False]
    slrc_new = slrc_new.rename(columns={'time': 'new_compilation'}, ).drop('slrc_is_old', axis=1)
    df = pandas.merge(slrc_old, slrc_new, on='name', how='inner')
    df = df[['name', 'old_compilation', 'new_compilation']]
    return df


if __name__ == '__main__':
    for domain in ['grid', 'zenotravel', 'blocksworld', 'driverlog']:

        df = pandas.read_csv(r'C:\Users\foree\PycharmProjects\up_social_laws_experimentation\experimentation\logs\experiment_log_transformed_Nov-01-2024.csv')
        df['name'] = df['name'].astype(str)  # Ensure column is of string type
        df = df.sort_values(by='name', key=lambda col: col.map(lambda x: int(re.search(r'\d+', x).group())))
        df['name'] = df['name'].str.replace('zenotravel_SL_', '')
        df = df.applymap(lambda x: '-' if isinstance(x, (int, float)) and x > 1800 else x)


        print(df)
        old_df = pandas.read_csv(r'C:\Users\foree\PycharmProjects\up_social_laws_experimentation\test\logs\experiment_log_Oct-31-2024_sl_no_sl_zenotravel.csv')
        old_df = old_df.sort_values(by='name', key=lambda col: col.map(lambda x: int(re.search(r'\d+', x).group())))
        print(old_df)
        old_df['old_compilation_with_sl'] = df['old_compilation']
        old_df['new_compilation_with_sl'] = df['new_compilation']
        print(old_df)
        old_df = old_df.applymap(lambda x: round(x,2) if isinstance(x, (int, float)) else x)

        #df[df.columns.difference(['name'])] = df[df.columns.difference(['name'])].round(3)

        latex_table = tabulate(old_df, headers='keys', tablefmt='latex', showindex=False)

        # Print or save the LaTeX table
        print(latex_table)

