import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split

# helper function 
def initialize_features(data): 
    # in this function we will turn the data into features array 
    pattern = r'Patient_(\S+)_Timepoint_(\d+)'
    patient_timepoints = []
    intensity_cols = [col for col in data.columns if re.search(pattern, col)]
    data[intensity_cols] = data[intensity_cols].replace({',': ''}, regex=True).apply(pd.to_numeric)
    data[intensity_cols] = data[intensity_cols].fillna(0)
    construct_data = {"patient_id": [], "timepoint": []}
    for i in range(len(data)):
        construct_data[f"protein_{i}"] = []
    for col in data.columns:
        match = re.search(pattern, col)
        if match:
            data[col] = data[col].astype(float)
            data[col] = data[col].fillna(0)
            # append the column data as new columns

            construct_data["patient_id"].append(match.group(1))
            construct_data["timepoint"].append(match.group(2))
            for i in range(len(data)):
                construct_data[f"protein_{i}"].append(data[col][i])
    # create a new dataframe with the patient_id and timepoint columns
    patient_timepoints_df = pd.DataFrame(construct_data)
    patient_timepoints_df.to_csv('data/patient_timepoints.csv', index=False)

    
def initialize_data():
    data = pd.read_csv('data/filtered_results.tsv', sep='\t', low_memory=False)
    unmod_columns = [col for col in data.columns if '_unmod' in col]
    data = data.drop(columns=unmod_columns)
    def rename_column(col_name):
    # Match the pattern '_dyn_#Patient_XX.Timepoint_Y'
        match = re.match(r'_dyn_#Patient_(\S+)\.Timepoint_(\d+)', col_name)
        if match:
            patient_id = match.group(1)
            timepoint = match.group(2)
            return f"Patient_{patient_id}_Timepoint_{timepoint}"
        return col_name

    data.columns = [rename_column(col) for col in data.columns]
    initialize_features(data)




def initialize_sample():
    # data_= data_pairwise_split()
    train_pairs = pd.read_csv('data/train_pair_keys.csv')
    val_pairs = pd.read_csv('data/val_pair_keys.csv')
    test_pairs = pd.read_csv('data/test_pair_keys.csv')

    ref_data = pd.read_csv('data/patient_timepoints.csv')

    trainset = train_pairs.sample(frac=0.1, random_state=42)[:50]
    valset = val_pairs.sample(frac=0.1, random_state=42)[:20]
    testset = test_pairs.sample(frac=0.1, random_state=42)[:20]

    # from ref_data get the protein data as a numpy array
    sample_dict = {}

    for i in range(len(ref_data)):
        patient_id = str(ref_data['patient_id'].iloc[i]).lstrip("0")
        timepoint = int(ref_data['timepoint'].iloc[i])
        # print(f"Patient ID: {patient_id}, Timepoint: {timepoint}")
        # if patient_id == "42":
            # print(f"We have 3 and {timepoint}")
            # print(ref_data.iloc[i, 2:10].to_numpy())
        sample_dict[(patient_id, timepoint)] = ref_data.iloc[i, 2:].to_numpy()

    train_data, train_labels = pair_to_feature(trainset, sample_dict)
    val_data, val_labels = pair_to_feature(valset, sample_dict)
    test_data, test_labels = pair_to_feature(testset, sample_dict)

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Val labels shape: {val_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # save the data to a numpy file
    np.savez('data/train_data.npz', data=train_data, labels=train_labels)
    np.savez('data/val_data.npz', data=val_data, labels=val_labels)
    np.savez('data/test_data.npz', data=test_data, labels=test_labels)

def make_pair_keys(left, right):
    """
    Given two small DataFrames `left` and `right` that each have
    only ['patient_id','timepoint'], return the 
    cross-product minus mirrors/self-pairs.
    """
    # do the cross-join on just the two key columns
    pairs = (
        left
        .merge(right, how='cross', suffixes=('_a','_b'))
        .query(
            "patient_id_a  <  patient_id_b"
            " or "
            "(patient_id_a == patient_id_b and timepoint_a < timepoint_b)"
        )
        .reset_index(drop=True)
    )
    return pairs

def data_pairwise_split(path='data/patient_timepoints.csv'):
    # 1) read and split patients
    df = pd.read_csv(path, dtype={'patient_id': str, 'timepoint': int})
    rng = np.random.RandomState(42)

    patients = df['patient_id'].unique()
    train_ids, temp_ids = train_test_split(patients, test_size=0.30, random_state=rng)
    val_ids, test_ids  = train_test_split(temp_ids, test_size=0.50, random_state=rng)

    # 2) slice out just the key columns
    keys = df[['patient_id','timepoint']]

    train_keys = keys[keys.patient_id.isin(train_ids)]
    val_keys   = keys[keys.patient_id.isin(val_ids)]
    test_keys  = keys[keys.patient_id.isin(test_ids)]

    # 3) build the pair-keys
    train_pairs = make_pair_keys(train_keys, train_keys)
    val_pairs   = make_pair_keys(val_keys, pd.concat([train_keys, val_keys],   ignore_index=True))
    test_pairs  = make_pair_keys(test_keys, pd.concat([train_keys, val_keys, test_keys], ignore_index=True))

    # 4) persist just the keys
    train_pairs.to_csv('data/train_pair_keys.csv', index=False)
    val_pairs.  to_csv('data/val_pair_keys.csv',   index=False)
    test_pairs. to_csv('data/test_pair_keys.csv',  index=False)

    return train_keys, val_keys, test_keys, train_pairs, val_pairs, test_pairs

def pair_to_feature(dataset, sample_dict):
    data = []
    labels = []
    for i in range(len(dataset)):
        a_id = str(dataset['patient_id_a'].iloc[i]).lstrip("0")
        b_id = str(dataset['patient_id_b'].iloc[i]).lstrip("0")
        a_timepoint = int(dataset['timepoint_a'].iloc[i])
        b_timepoint = int(dataset['timepoint_b'].iloc[i])
        vec_a = sample_dict[(a_id, a_timepoint)]
        vec_b = sample_dict[(b_id, b_timepoint)]
        data.append(np.abs(vec_a - vec_b))
        labels.append(1 if a_id == b_id else 0)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels