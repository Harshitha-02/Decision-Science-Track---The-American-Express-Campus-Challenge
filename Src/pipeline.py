import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')

def process_train_data(df):
    drops = ['match id', 'team1', 'team1_id', 'team1_roster_ids', 'team2', 'team2_id', 'team2_roster_ids',
             'winner_id', 'venue', 'city', 'ground_id', 'match_dt', 'series_name', 'season']
    
    df['winner'] = np.where(df['winner'] == df['team1'], 0, 1)
    df['toss winner'] = np.where(df['toss winner'] == df['team1'], 0, 1)
    df['toss decision'] = np.where(df['toss decision'] == 'field', 0, 1)
    df = pd.get_dummies(df, columns=['lighting'])
    df.drop(drops, axis=1, inplace=True)
    df = df.astype(int, errors='ignore')

    return df

def merge_df(df_train, df_bat, df_bowl, df_match):
    df = pd.merge(df_bat, df_bowl, on='match_id', how='inner')
    df = pd.merge(df, df_match, on='match_id', how='inner')
    return df