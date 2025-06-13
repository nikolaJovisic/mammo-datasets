import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("rsna.csv")

# Drop rows with missing rad_recall
df = df.dropna(subset=["cancer"])

# Get unique patients and their rad_recall label (use majority label if duplicated)
patient_labels = (
    df.groupby("patient_id")["cancer"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# Split patients by class
patients_0 = patient_labels[patient_labels["cancer"] == 0]
patients_1 = patient_labels[patient_labels["cancer"] == 1]

# Helper function to split patient IDs
def stratified_patient_split(patients, train_ratio=0.7, valid_ratio=0.15):
    train_ids, temp_ids = train_test_split(
        patients["patient_id"], train_size=train_ratio, random_state=42
    )
    valid_ids, test_ids = train_test_split(
        temp_ids, train_size=valid_ratio / (1 - train_ratio), random_state=42
    )
    return set(train_ids), set(valid_ids), set(test_ids)

# Split both classes
train_0, valid_0, test_0 = stratified_patient_split(patients_0)
train_1, valid_1, test_1 = stratified_patient_split(patients_1)

# Combine and label all patient IDs
train_ids = train_0 | train_1
valid_ids = valid_0 | valid_1
test_ids = test_0 | test_1

# Assign split labels
def assign_split(row):
    pid = row["patient_id"]
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

# Save to files
df[df["split"] == "train"].to_csv("rsna_train.csv", index=False)
df[df["split"] == "valid"].to_csv("rsna_valid.csv", index=False)
df[df["split"] == "test"].to_csv("rsna_test.csv", index=False)
