import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("vindr_breast_level.csv")
df = df.dropna(subset=["breast_birads"])

patient_labels = (
    df.groupby("study_id")["breast_birads"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# Get all unique classes
classes = patient_labels["breast_birads"].unique()

# Split per class
train_ids, valid_ids, test_ids = set(), set(), set()

for cls in classes:
    patients_cls = patient_labels[patient_labels["breast_birads"] == cls]
    train_cls, temp_cls = train_test_split(
        patients_cls["study_id"], train_size=0.7, random_state=42, stratify=None
    )
    valid_cls, test_cls = train_test_split(
        temp_cls, train_size=0.5, random_state=42, stratify=None
    )
    train_ids |= set(train_cls)
    valid_ids |= set(valid_cls)
    test_ids |= set(test_cls)

def assign_split(row):
    pid = row["study_id"]
    if pid in train_ids:
        return "train"
    elif pid in valid_ids:
        return "valid"
    elif pid in test_ids:
        return "test"
    else:
        return None

df["split"] = df.apply(assign_split, axis=1)
df = df[df["split"].notnull()]

df[df["split"] == "train"].to_csv("vindr_train.csv", index=False)
df[df["split"] == "valid"].to_csv("vindr_valid.csv", index=False)
df[df["split"] == "test"].to_csv("vindr_test.csv", index=False)
