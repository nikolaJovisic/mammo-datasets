import pandas as pd

column_name = 'label'


#csaw
csaw = pd.read_csv('csaw.csv')

csaw[column_name] = csaw['rad_recall']
csaw.loc[
    (csaw['rad_recall'] == 0.0) & ((csaw['rad_r1'] == 1.0) | (csaw['rad_r2'] == 1.0)),
    column_name
] = 2.0

csaw.to_csv('csaw.csv', index=False)



#rsna

rsna = pd.read_csv('rsna.csv')

rsna[column_name] = None

# 3 - biopsy proven cancer
rsna.loc[rsna['cancer'] == 1, column_name] = 3

# 2 - difficult negative case
rsna.loc[
    (rsna[column_name].isna()) &
    (rsna['difficult_negative_case'] == True),
    column_name
] = 2

# 1 - no cancer
rsna.loc[
    (rsna[column_name].isna()) &
    (rsna['BIRADS'] == 1),
    column_name
] = 1

# 0 - negative
rsna.loc[
    (rsna[column_name].isna()) &
    (rsna['BIRADS'] == 2),
    column_name
] = 0

rsna.to_csv('rsna.csv', index=False)
