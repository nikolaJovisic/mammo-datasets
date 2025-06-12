import pandas as pd

def aggregate_by_breast(df, label_col='asses', 
                            view_col='ViewPosition', 
                            laterality_col='ImageLateralityFinal', 
                            study_col='acc_anon', 
                            path_col='png_path', 
                            cc_col='CC', 
                            mlo_col='MLO'):
    
    df = df.dropna(subset=[label_col])
    df = df[df[view_col].isin([cc_col, mlo_col])]

    paths = df.pivot_table(
        index=[study_col, laterality_col],
        columns=view_col,
        values=path_col,
        aggfunc='first'
    ).reset_index()

    label = df.groupby([study_col, laterality_col])[label_col].max().reset_index()
    df = pd.merge(paths, label, on=[study_col, laterality_col])

    for col in [cc_col, mlo_col]:
        df[col] = df[col].where(df[col].notna(), None).astype(object)
        
    return df