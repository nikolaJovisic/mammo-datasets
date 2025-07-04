import pandas as pd

def aggregate_by_breast(df, label_col, laterality_col, study_col, path_col):
    df = df.dropna(subset=[label_col])
    paths = df.groupby([study_col, laterality_col])[path_col].apply(list).reset_index()
    df['window'] = list(zip(df['window_a'], df['window_b']))
    windowings = df.groupby([study_col, laterality_col])['window'].apply(list).reset_index()
    label = df.groupby([study_col, laterality_col])[label_col].max().reset_index()
    df = pd.merge(paths, windowings, on=[study_col, laterality_col])
    df = pd.merge(df, label, on=[study_col, laterality_col])
    return df
