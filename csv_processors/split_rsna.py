import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("rsna.csv")
df = df.dropna(subset=["cancer"])

patient_labels = (
    df.groupby("patient_id")["cancer"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

train_ids, valid_ids, test_ids = set(), set(), set()

for label in [0, 1]:
    patients = patient_labels[patient_labels["cancer"] == label]
    train_split, temp_split = train_test_split(
        patients["patient_id"], train_size=0.7, random_state=42
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

df["split"] = df["patient_id"].apply(assign_split)
df = df[df["split"].notnull()]

df[df["split"] == "train"].to_csv("rsna_train.csv", index=False)
df[df["split"] == "valid"].to_csv("rsna_valid.csv", index=False)
df[df["split"] == "test"].to_csv("rsna_test.csv", index=False)
