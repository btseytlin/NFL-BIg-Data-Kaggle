import pandas as pd
import numpy as np
from string import punctuation
import datetime
import re

def crps(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=1)

def yard_to_cdf(yard):
    yard = np.round(yard).astype(int)
    indices = yard+99
    cdfs = np.zeros((yard.shape[0], 199))
    for i in range(len(cdfs)):
        cdfs[i, indices[i]:] = 1
    return cdfs

def cdf_to_yard(cdf):
    yard_index = (cdf==1).argmax(axis=1)
    yard = yard_index-99
    return yard

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
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
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
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle

def preprocess_features(df, fillna=False):
    """Accepts df like train data, returns X, y"""

    # Feature engineering
    df['DefendersInTheBox_vs_Distance'] = (df['DefendersInTheBox'] / df['Distance'])
    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
    df['StadiumTypeShort'] = df['StadiumType'].apply(transform_StadiumType)
    df = df.drop(['StadiumType'], axis=1)

    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

    turf_type = df['Turf'].map(Turf)
    df['TurfIsNatural'] = (turf_type == 'Natural')
    df = df.drop(['Turf'], axis=1)

    # CAREFUL. What if a new team appears?
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb

    def safe_map(val):
        if map_abbr.get('val'):
            return map_abbr[val]
        else:
            return val

    df['PossessionTeam'] = df['PossessionTeam'].apply(safe_map)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].apply(safe_map)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].apply(safe_map)

    df['HomePossesion'] = (df['PossessionTeam'] == df['HomeTeamAbbr'])

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
    df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']

    # in posession
    df['InPosession']=(((df.Team == 'home') & (df.PossessionTeam == df.HomeTeamAbbr)) | ((df.Team == 'away') & (df.PossessionTeam == df.VisitorTeamAbbr)))

    # Formation columns
    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='OffenseFormation')], axis=1)
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


    df['GameClock'] = df['GameClock'].apply(str_to_seconds)

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight'])**2)

    df['TimeHandoff'] = pd.to_datetime(df['TimeHandoff'], utc=True)
    df['TimeSnap'] = pd.to_datetime(df['TimeSnap'], utc=True)
    df['TimeDelta'] = (df['TimeHandoff']-df['TimeSnap']).apply(lambda x: x.total_seconds())
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'], utc=True)

    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = (df['TimeHandoff']-df['PlayerBirthDate']).apply(lambda x: x.total_seconds())/seconds_in_year
    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    df = df.drop(['WindDirection'], axis=1)
    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')
    df['IsHomeTeam'] = df['Team'].apply(lambda x: x.strip()=='home')



    df['GameWeather'] = df['GameWeather'].str.lower()
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)

    df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)
    
    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
    

    # DefensePersonnel
    counts = []
    for i, val in df['DefensePersonnel'].str.split(',').iteritems():
        row = {'OL': 0, 'RB': 0, 'TE': 0, 'WR': 0, 'DL': 0, 'DB': 0, 'LB': 0, 'QB': 0}
        if val is np.NaN:
            counts.append({})
            continue
        for item in val:
            name, number = item.strip().split(' ')[::-1]
            row[name] = int(number)
        counts.append(row)
    defense_presonell_df = pd.DataFrame(counts)
    defense_presonell_df.columns = ['defense_'+x for x in defense_presonell_df.columns]
    defense_presonell_df = defense_presonell_df.fillna(0).astype(int)
    defense_presonell_df.index = df.index
    df = pd.concat([df.drop(['DefensePersonnel'], axis=1), defense_presonell_df], axis=1)


    # OffensePersonnel
    counts = []
    for i, val in df['OffensePersonnel'].str.split(',').iteritems():
        row = {'OL': 0, 'RB': 0, 'TE': 0, 'WR': 0, 'DL': 0, 'DB': 0, 'LB': 0, 'QB': 0}
        if val is np.NaN:
            counts.append({})
            continue
        for item in val:
            name, number = item.strip().split(' ')[::-1]
            row[name] = int(number)
        counts.append(row)
    offense_personnel_df = pd.DataFrame(counts)
    offense_personnel_df.columns = ['offense_'+x for x in offense_personnel_df.columns]
    offense_personnel_df = offense_personnel_df.fillna(0).astype(int)
    offense_personnel_df.index = df.index
    df = pd.concat([df.drop(['OffensePersonnel'], axis=1), offense_personnel_df], axis=1)
    

    df = sort_df(df)

    if fillna:
        df.fillna(-999, inplace=True)
    return df

def sort_df(df):
    df = df.sort_values(by=['PlayId', 'InPosession', 'IsRusher']).reset_index(drop=True)
    return df

def make_x(df):
    source_play_id = df['PlayId']

    cols_delete = ['GameId', 'PlayId', 'IsRusher', 'Team']
    df = df.drop(cols_delete, axis=1)

    # Fill nan

    # Text features
    text_cols = [
        'FieldPosition',
         'DisplayName',
         'PossessionTeam',
         'PlayerCollegeName',
         'Position',
         'HomeTeamAbbr',
         'VisitorTeamAbbr',
         'Stadium',
         'Location']
    df = df.drop(text_cols, axis=1)

    # Player features
    cols_player = ['X',
         'Y',
         'S',
         'A',
         'Dis',
         'Orientation',
         'Dir',
         'JerseyNumber',
         'PlayerHeight',
         'PlayerWeight',
         'PlayerBMI',
         'PlayerAge']

    all_cols_player = np.array([[f'pl{num}_'+x for x in cols_player] for num in range(1, 23)]).flatten()

    X = np.array(df[cols_player]).reshape(-1, len(cols_player)*22)

    play_id_index = source_play_id[::22]
    X_df = pd.DataFrame(X, columns=all_cols_player, index=play_id_index)

    assert df[cols_player].shape[0] == X_df.shape[0] * 22
    assert df[cols_player].shape[1] == X_df.shape[1] / 22

    # Play features
    cols_play = list(df.drop(cols_player+(['Yards'] if 'Yards' in df.columns else []), axis=1).columns)
    X_play_col = np.zeros(shape=(X.shape[0], len(cols_play)))
    for i, col in enumerate(cols_play):
        X_play_col[:, i] = df[col][::22]

    X_play_col_df = pd.DataFrame(X_play_col, columns=cols_play, index=play_id_index)
    assert X_df.shape[0] == X_play_col_df.shape[0]
    X_df = pd.concat([X_df, X_play_col_df], axis=1)

    assert X_df.shape[0] == source_play_id.drop_duplicates().count()
    return X_df

def make_y(X, df):
    y = np.zeros(shape=(X.shape[0], 199))
    for i, yard in enumerate(df['Yards'][::22]):
        y[i, yard+99:] = np.ones(shape=(1, 100-yard))
    return y

