import pandas as pd
import numpy as np

def extract_per_key_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Przekształca surowe dane w macierz per-klawisz, automatycznie 
    filtrując anomalie czasowe.
    """
    df = df_raw.copy()
    
    df['KeyPressed'] = df['KeyPressed'].str.lower()
    
    df['DwellTime'] = df['ReleaseTime'] - df['PressTime']
    
    df = df[(df['DwellTime'] > 0) & (df['DwellTime'] < 1000)]
    
    allowed_keys = list("abcdefghijklmnopqrstuvwxyz ")
    df = df[df['KeyPressed'].isin(allowed_keys)]
    
    features = df.pivot_table(
        index=['UserId', 'SampleNumber'], 
        columns='KeyPressed', 
        values='DwellTime', 
        aggfunc='mean'
    )
    
    return features.fillna(0).reset_index()