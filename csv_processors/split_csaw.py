import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("csaw.csv")
df = df.dropna(subset=["rad_recall"])

patient_labels = (
    df.groupby("anon_patientid")["rad_recall"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

train_ids, valid_ids, test_ids = set(), set(), set()

for label in [0, 1]:
    patients = patient_labels[patient_labels["rad_recall"] == label]
    train_split, temp_split = train_test_split(
        patients["anon_patientid"], train_size=0.7, random_state=42
    )
    valid_split, test_split = train_test_split(
        temp_split, train_size=0.5, random_state=42
    )
    train_ids |= set(train_split)
    valid_ids |= set(valid_split)
    test_ids |= set(test_split)

def assign_split(pid):
    if pid in train_ids:
        return "train"
    elif pid in valid_ids:
        return "valid"
    elif pid in test_ids:
        return "test"
    else:
        return None

df["split"] = df["anon_patientid"].apply(assign_split)
df = df[df["split"].notnull()]

condition = (df["rad_recall"] == 0.0) & ((df["rad_r1"] == 1.0) | (df["rad_r2"] == 1.0))
df.loc[condition, "split"] = "test"
df.loc[condition, "rad_recall"] = 2.0

df[df["split"] == "train"].to_csv("csaw_train.csv", index=False)
df[df["split"] == "valid"].to_csv("csaw_valid.csv", index=False)
df[df["split"] == "test"].to_csv("csaw_test.csv", index=False)
