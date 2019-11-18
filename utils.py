import pandas as pd
import numpy as np
from string import punctuation
import datetime
import re
import torch
import math
from tqdm.notebook import tqdm

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                    np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                    np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                    np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                    np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                    np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                    np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f'Mem. usage decreased to {end_mem} Mb ({100 * (start_mem - end_mem) / start_mem}% reduction)')
    return df


def crps(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=1)


def yard_to_cdf(yard):
    yard = np.round(yard).astype(int)
    indices = yard + 99
    cdfs = np.zeros((yard.shape[0], 199))
    for i in range(len(cdfs)):
        cdfs[i, indices[i]:] = 1
    return cdfs


def cdf_to_yard(cdf):
    yard_index = (cdf == 1).argmax(axis=1)
    yard = yard_index - 99
    return yard


def cdf_to_yard_torch(cdf):
    yard_index = torch.sum((torch.as_tensor(cdf) <= 0), dim=1)
    yard = yard_index - 99
    return yard


def crps_torch(y_true, y_pred):
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    return torch.mean((y_true - y_pred).pow(2), dim=1)


def crps_loss(y_true, y_pred_pdf):
    y_pred_cdf = torch.cumsum(torch.as_tensor(y_pred_pdf), dim=1)
    return crps_torch(y_true, y_pred_cdf).mean()


def crps_loss_cdf(y_true, y_pred_cdf):
    return crps_torch(y_true, y_pred_cdf).mean()


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    return txt


def transform_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0
    return np.nan


def str_to_seconds(txt):
    txt = txt.split(':')
    ans = int(txt[0]) * 60 + int(txt[1]) + int(txt[2]) / 60
    return ans


def str_to_float(txt):
    try:
        return float(txt)
    except Exception as e:
        return np.NaN


def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans *= 0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans * 3
    if 'sunny' in txt or 'sun' in txt:
        return ans * 2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2 * ans
    if 'snow' in txt:
        return -3 * ans
    return 0


def standartize_orientations(df):
    """
        https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python

        Make sure the offensive team is always moving left to right.
    """
    df['ToLeft'] = df.PlayDirection == "left"
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense  # Is player on offense?
    df['HomeOnOffense'] = (df['TeamOnOffense'] == 'home').astype(int)

    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
           'YardLine_std'
    ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
               'YardLine']
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160 / 3 - df.loc[df.ToLeft, 'Y']

    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi / 180.0
    df['Dir_std'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'],
                                          2 * np.pi)

    df['Orientation_rad'] = np.mod(df.Orientation, 360) * math.pi / 180.0

    df.loc[df.Season >= 2018, 'Orientation_rad'
    ] = np.mod(df.loc[df.Season >= 2018, 'Orientation'] - 90,
               360) * math.pi / 180.0

    df['Orientation_rad'] = np.mod(df.Orientation, 360) * math.pi / 180.0
    df.loc[df.Season >= 2018, 'Orientation_rad'
    ] = np.mod(df.loc[df.Season >= 2018, 'Orientation'] - 90,
               360) * math.pi / 180.0
    df['Orientation_std'] = df.Orientation_rad
    df.loc[df.ToLeft, 'Orientation_std'] = np.mod(
        math.pi + df.loc[df.ToLeft, 'Orientation_rad'], 2 * math.pi)

    replace_cols = ['YardLine', 'X', 'Y', 'Dir', 'Orientation']
    for col in replace_cols:
        df[col] = df[col + '_std']
        df.drop([col + '_std'], axis=1, inplace=True)

    drop_cols = ['Dir_rad', 'Orientation_rad']
    for col in drop_cols:
        df.drop([col], axis=1, inplace=True)

    return df


def map_turf(df):
    Turf = {'Field Turf': 'Artificial', 'A-Turf Titan': 'Artificial',
            'Grass': 'Natural', 'UBU Sports Speed S5-M': 'Artificial',
            'Artificial': 'Artificial', 'DD GrassMaster': 'Artificial',
            'Natural Grass': 'Natural',
            'UBU Speed Series-S5-M': 'Artificial', 'FieldTurf': 'Artificial',
            'FieldTurf 360': 'Artificial', 'Natural grass': 'Natural',
            'grass': 'Natural',
            'Natural': 'Natural', 'Artifical': 'Artificial',
            'FieldTurf360': 'Artificial', 'Naturall Grass': 'Natural',
            'Field turf': 'Artificial',
            'SISGrass': 'Artificial', 'Twenty-Four/Seven Turf': 'Artificial',
            'natural grass': 'Natural'}
    
    turf_type = df['Turf'].map(Turf)
    df['TurfIsNatural'] = (turf_type == 'Natural')
    df = df.drop(['Turf'], axis=1)
    return df

def clean_abbrs(df):
    # CAREFUL. What if a new team appears?
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb

    def safe_map(val):
        if map_abbr.get(val):
            return map_abbr[val]
        else:
            return val

    df['PossessionTeam'] = df['PossessionTeam'].apply(safe_map)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].apply(safe_map)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].apply(safe_map)
    df['FieldPosition'] = df['FieldPosition'].apply(safe_map)
    return df

def clean_weather(df):
    df['WindSpeed'] = df['WindSpeed'].apply(
        lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0]) + int(
        x.split('-')[1])) / 2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(
        lambda x: (int(x.split()[0]) + int(x.split()[-1])) / 2 if not pd.isna(
            x) and type(x) != float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    df = df.drop(['WindDirection'], axis=1)

    df['GameWeather'] = df['GameWeather'].str.lower()
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(
        lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(
        lambda x: x.replace('coudy', 'cloudy').replace('clouidy',
                                                       'cloudy').replace('party',
                                                                         'partly') if not pd.isna(
            x) else x)
    df['GameWeather'] = df['GameWeather'].apply(
        lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(
            x) else x)
    df['GameWeather'] = df['GameWeather'].apply(
        lambda x: x.replace('skies', '').replace("mostly",
                                                 "").strip() if not pd.isna(
            x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)
    return df

# Encoding

def encode_formations(df):
    # Formation columns
    df = pd.concat([df.drop(['OffenseFormation'], axis=1),
                    pd.get_dummies(df['OffenseFormation'],
                                   prefix='OffenseFormation')], axis=1)
    # Filling missing dummy columns at test stage
    expected_columns = ['OffenseFormation_ACE',
                        'OffenseFormation_EMPTY',
                        'OffenseFormation_JUMBO',
                        'OffenseFormation_PISTOL',
                        'OffenseFormation_SHOTGUN',
                        'OffenseFormation_SINGLEBACK',
                        'OffenseFormation_WILDCAT',
                        'OffenseFormation_I_FORM']
    for col in expected_columns:
        if not col in df.columns:
            df[col] = 0
    return df

def encode_personell(df):
    # DefensePersonnel
    counts = []
    for i, val in df['DefensePersonnel'].str.split(',').iteritems():
        row = {'OL': 0, 'RB': 0, 'TE': 0, 'WR': 0, 'DL': 0, 'DB': 0, 'LB': 0,
               'QB': 0}
        if val is np.NaN:
            counts.append({})
            continue
        for item in val:
            name, number = item.strip().split(' ')[::-1]
            row[name] = int(number)
        counts.append(row)
    defense_presonell_df = pd.DataFrame(counts)
    defense_presonell_df.columns = ['defense_' + x for x in
                                    defense_presonell_df.columns]
    defense_presonell_df = defense_presonell_df.fillna(0).astype(int)
    defense_presonell_df.index = df.index
    df = pd.concat([df.drop(['DefensePersonnel'], axis=1), defense_presonell_df],
                   axis=1)

    # OffensePersonnel
    counts = []
    for i, val in df['OffensePersonnel'].str.split(',').iteritems():
        row = {'OL': 0, 'RB': 0, 'TE': 0, 'WR': 0, 'DL': 0, 'DB': 0, 'LB': 0,
               'QB': 0}
        if val is np.NaN:
            counts.append({})
            continue
        for item in val:
            name, number = item.strip().split(' ')[::-1]
            row[name] = int(number)
        counts.append(row)
    offense_personnel_df = pd.DataFrame(counts)
    offense_personnel_df.columns = ['offense_' + x for x in
                                    offense_personnel_df.columns]
    offense_personnel_df = offense_personnel_df.fillna(0).astype(int)
    offense_personnel_df.index = df.index
    df = pd.concat([df.drop(['OffensePersonnel'], axis=1), offense_personnel_df],
                   axis=1)
    return df

def engineer_features(df):
    df['DefendersInTheBox_vs_Distance'] = (
        df['DefendersInTheBox'] / df['Distance'])

    df = add_phys_features(df)
    return df

def preprocess_features(df):
    """Accepts df like train data, returns cleaned, standartized and enriched df"""

    df = clean_abbrs(df)

    df = standartize_orientations(df)

    df = map_turf(df)

    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
    df['StadiumTypeShort'] = df['StadiumType'].apply(transform_StadiumType)
    df = df.drop(['StadiumType'], axis=1)

    # df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']
    # df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
    # df['HomePossesion'] = (df['PossessionTeam'] == df['HomeTeamAbbr'])

    df['GameClock'] = df['GameClock'].apply(str_to_seconds)

    df['PlayerHeight'] = df['PlayerHeight'].apply(
        lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))
    df['PlayerBMI'] = 703 * (df['PlayerWeight'] / (df['PlayerHeight']) ** 2)

    df['TimeHandoff'] = pd.to_datetime(df['TimeHandoff'], utc=True)
    df['TimeSnap'] = pd.to_datetime(df['TimeSnap'], utc=True)
    df['TimeDelta'] = (df['TimeHandoff'] - df['TimeSnap']).apply(
        lambda x: x.total_seconds())
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(
        lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'], utc=True)

    seconds_in_year = 60 * 60 * 24 * 365.25
    df['PlayerAge'] = (df['TimeHandoff'] - df['PlayerBirthDate']).apply(
        lambda x: x.total_seconds()) / seconds_in_year

    df = clean_weather(df)

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    df['IsQB'] = df['Position'] == 'QB'

    df = sort_df(df)

    df = reduce_mem_usage(df)

    return df

def sort_df(df):
    df.sort_values(by=['PlayId', 'IsOnOffense', 'IsRusher', 'IsQB'], inplace=True)
    return df

def make_y(df):
    y = np.zeros(shape=(df['Yards'][::22].shape[0], 199))
    for i, yard in enumerate(df['Yards'][::22]):
        y[i, yard + 99:] = np.ones(shape=(1, 100 - yard))
    return y

def compute_play_phys_features(play_id, play_df):
    assert play_df.shape[0] == 22
    play_index = play_df.index
    offense_index = play_df.IsOnOffense
    defense_index = ~play_df.IsOnOffense
    offense_df = play_df[offense_index]
    defense_df = play_df[defense_index]

    offense_centroid_x = offense_df.X.mean()
    offense_centroid_y = offense_df.Y.mean()
    offense_x_std = offense_df.X.std()
    offense_y_std = offense_df.Y.std()
    offense_mean_force = offense_df.Force.mean()

    defense_centroid_x = defense_df.X.mean()
    defense_x_std = defense_df.X.std()
    defense_centroid_y = defense_df.Y.mean()
    defense_y_std = defense_df.Y.std()
    defense_mean_force = defense_df.Force.mean()

    rusher_row = offense_df[offense_df.IsRusher].iloc[0]

    qb_row = offense_df[offense_df.Position == 'QB']
    if not qb_row.empty:
        qb_row = qb_row.iloc[0]

    offense_centroid_pos = np.array([offense_centroid_x, offense_centroid_y])
    defense_centroid_pos = np.array([defense_centroid_x, defense_centroid_y])
    rusher_pos = rusher_row[['X', 'Y']].values[0]
    qb_pos = qb_row[['X', 'Y']].values[0] if not qb_row.empty else None

    rusher_dist_to_qb = np.linalg.norm(
        rusher_pos - qb_pos) if not qb_row.empty else None
    rusher_dist_to_offense_centroid = np.linalg.norm(
        rusher_pos - offense_centroid_pos)
    rusher_dist_to_defence_centroid = np.linalg.norm(
        rusher_pos - defense_centroid_pos)

    # Defender to rusher distances
    defense_distances_to_runner = []
    for row in defense_df.itertuples():
        pos = np.array([row.X, row.Y])
        defense_distances_to_runner.append(np.linalg.norm(rusher_pos - pos))
    dist_to_rusher = defense_distances_to_runner = np.array(
        defense_distances_to_runner)

    time_to_rusher = defense_df['S'] / dist_to_rusher

    defender_dist_to_runner_min = defense_distances_to_runner.min()

    defender_time_to_runner_min = time_to_rusher.min()

    # closest defenders
    closest_defender = \
        defense_df[dist_to_rusher == defender_dist_to_runner_min].iloc[0]
    closest_bytime_defender = \
        defense_df[time_to_rusher == defender_time_to_runner_min].iloc[0]

    closest_defender_force_div_rusher_force = (
        closest_defender.Force / rusher_row.Force)
    closest_bytime_defender_force_div_rusher_force = (
        closest_bytime_defender.Force / rusher_row.Force)

    closest_bytime_defender_speed_div_rusher_speed = closest_bytime_defender.S / rusher_row.S
    closest_bytime_defender_acceleration_div_rusher_acceleration = closest_bytime_defender.A / rusher_row.A

    # Add play features
    new_play_features = {
        'phys_offense_x_std': offense_x_std,
        'phys_offense_y_std': offense_y_std,
        'phys_offense_mean_force': offense_mean_force,
        'phys_defense_x_std': defense_x_std,
        'phys_defense_y_std': defense_y_std,
        'phys_defense_mean_force': defense_mean_force,

        'phys_rusher_dist_to_qb': rusher_dist_to_qb,
        'phys_rusher_dist_to_offense_centroid': rusher_dist_to_offense_centroid,
        'phys_rusher_dist_to_defence_centroid': rusher_dist_to_defence_centroid,

        'phys_defender_dist_to_runner_min': defender_dist_to_runner_min,

        'phys_defender_time_to_runner_min': defender_time_to_runner_min,

        'phys_closest_defender_force_div_rusher_force': closest_defender_force_div_rusher_force,
        'phys_closest_bytime_defender_force_div_rusher_force': closest_bytime_defender_force_div_rusher_force,

        'phys_closest_bytime_defender_speed_div_rusher_speed': closest_bytime_defender_speed_div_rusher_speed,
        'phys_closest_bytime_defender_acceleration_div_rusher_acceleration': closest_bytime_defender_acceleration_div_rusher_acceleration,
    }
    return new_play_features

def compute_phys_features(df):
    playid_index = df.PlayId.unique()
    features_rows= []
    for play_id in tqdm.tqdm(playid_index):
        play_df = df[df.PlayId == play_id]
        features_rows.append(compute_play_phys_features(play_id, play_df))

    features_df = pd.DataFrame(features_rows, index=playid_index)
    return features_df

def make_x(df, encode=True, fillna=True, verbose=True):
    """Input: source data, preprocesed.
       Output: X dataframe
    """
    source_play_id = df['PlayId']
    df = df.copy()
    df.index = source_play_id

    # Encoding
    if encode:
        df = encode_formations(df)
        df = encode_personell(df)

    # Feature engineering

    df['DefendersInTheBox_vs_Distance'] = (
        df['DefendersInTheBox'] / df['Distance'])

    df['Force'] = df['A'] * df['PlayerWeight']

    if verbose:
        print('Computing phys features')

    phys_features = compute_phys_features(df)

    # Assemble X

    # Drop unnececary rows, keep only rusher

    df = df[df.IsRusher].copy()

    # Drop unnececary columns

    cols_to_drop = [
        'Yards', 'PlayDirection', 'TeamOnOffense', 'NflId', 'NflIdRusher',
        'TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'FieldPosition',
        'DisplayName', 'PossessionTeam', 'PlayerCollegeName', 'Position',
        'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location', 'GameId',
        'PlayId', 'Team', 'IsRusher', 'IsQB', 'IsOnOffense', 'Toleft',
        'HomeOnOffense', 'Temperature', 'Humidity', 'WindSpeed', 'GameWeather',
        'WindDirection', 'ToLeft', 'X', 'Y', 'Dis', 'PlayerWeight', 'PlayerHeight'
    ]
    cols_to_drop = list(set(cols_to_drop).intersection(set(list(df.columns))))
    df.drop(cols_to_drop, axis=1, inplace=True)

    if verbose:
        print('Dropped cols:', cols_to_drop)

    # Assemble X
    
    # Get player columns
    cols_player = ['S',
                   'A',
                   'Orientation',
                   'Dir',
                   'PlayerBMI',
                   'PlayerAge',
                   'Force']
                
    X_df = df.copy()

    # Add features

    X_df = pd.concat([X_df, phys_features], axis=1)

    # Pospreprocesing

    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    if fillna:
        X_df.fillna(-999, inplace=True)

    X_df = reduce_mem_usage(X_df)

    assert X_df.shape[0] == source_play_id.drop_duplicates().count()
    return X_df
