import pandas as pd
import numpy as np
import os
from pathlib import Path


def convert_to_seen_split_simple(data_dir, split_index=1, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    seenMinute，7:2:1Minute，CPD_NAME on trainval

    Parameter:
    ----------
    data_dir : str
        DirectoryPATH
    split_index : int
        Minute（1）
    train_ratio : float
        Training（0.7）
    val_ratio : float
        Validation（0.2）
    seed : int
        random seed
    """

    np.random.seed(seed)


    train_path = Path(data_dir) / f"datasplit{split_index}-train.csv"
    val_path = Path(data_dir) / f"datasplit{split_index}-val.csv"
    test_path = Path(data_dir) / f"datasplit{split_index}-test.csv"


    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("Minute:")
    print("=" * 50)


    for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if 'CPD_NAME' in df.columns:
            unique_cpds = df['CPD_NAME'].nunique()
        else:

            possible_names = ['cpd_name', 'compound', 'treatment', 'perturbation']
            unique_cpds = 0
            for col in possible_names:
                if col in df.columns:
                    unique_cpds = df[col].nunique()
                    df['CPD_NAME'] = df[col]
                    break

        print(f"{name}: {len(df)}  samples, {unique_cpds} CPD_NAME")


    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)


    if 'CPD_NAME' not in all_data.columns:
        raise ValueError("CPD_NAME，")

    print(f"\n: {len(all_data)}  samples, {all_data['CPD_NAME'].nunique()} CPD_NAME")


    new_train = []
    new_val = []
    new_test = []

    unique_cpds = all_data['CPD_NAME'].unique()
    print(f"\nStartCPD_NAMEMinute...")

    for cpd in unique_cpds:
        cpd_data = all_data[all_data['CPD_NAME'] == cpd].copy()
        n_samples = len(cpd_data)

        if n_samples <= 2:

            if n_samples == 1:
                new_train.append(cpd_data)
            else:  # n_samples == 2

                shuffled = cpd_data.sample(frac=1, random_state=seed)
                new_train.append(shuffled.iloc[[0]])
                new_val.append(shuffled.iloc[[1]])
            continue


        cpd_data = cpd_data.sample(frac=1, random_state=seed)


        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))
        n_test = n_samples - n_train - n_val


        if n_test < 0:

            n_test = 0
            n_train = max(1, n_samples - 1)
            n_val = n_samples - n_train


        train_idx = n_train
        val_idx = n_train + n_val

        new_train.append(cpd_data.iloc[:train_idx])
        new_val.append(cpd_data.iloc[train_idx:val_idx])
        if n_test > 0:
            new_test.append(cpd_data.iloc[val_idx:])


    new_train_df = pd.concat(new_train, ignore_index=True)
    new_val_df = pd.concat(new_val, ignore_index=True)
    new_test_df = pd.concat(new_test, ignore_index=True) if new_test else pd.DataFrame(columns=all_data.columns)

    print("\nseenMinute:")
    print("=" * 50)
    print(f"Training: {len(new_train_df)}  samples, {new_train_df['CPD_NAME'].nunique()} CPD_NAME")
    print(f"Validation: {len(new_val_df)}  samples, {new_val_df['CPD_NAME'].nunique()} CPD_NAME")
    print(f"Test: {len(new_test_df)}  samples, {new_test_df['CPD_NAME'].nunique()} CPD_NAME")


    train_cpds = set(new_train_df['CPD_NAME'].unique())
    val_cpds = set(new_val_df['CPD_NAME'].unique())

    print("\n:")
    print("=" * 50)


    missing_in_val = train_cpds - val_cpds
    if len(missing_in_val) > 0:
        print(f"Warning: {len(missing_in_val)} CPD_NAME on Validation")
        print(f"CPD_NAME: {list(missing_in_val)[:5]}")
    else:
        print("✓ CPD_NAME on TrainingValidation")


    missing_in_train = val_cpds - train_cpds
    if len(missing_in_train) > 0:
        print(f"Warning: {len(missing_in_train)} CPD_NAME on Training")
        print(f"CPD_NAME: {list(missing_in_train)[:5]}")
    else:
        print("✓ CPD_NAME on Training")


    output_dir = Path(data_dir)
    new_train_path = output_dir / f"datasplit{split_index}-train-seen.csv"
    new_val_path = output_dir / f"datasplit{split_index}-val-seen.csv"
    new_test_path = output_dir / f"datasplit{split_index}-test-seen.csv"

    new_train_df.to_csv(new_train_path, index=False)
    new_val_df.to_csv(new_val_path, index=False)
    new_test_df.to_csv(new_test_path, index=False)

    print(f"\nMinuteFileSave:")
    print(f"  Training: {new_train_path}")
    print(f"  Validation: {new_val_path}")
    print(f"  Test: {new_test_path}")

    return new_train_df, new_val_df, new_test_df



if __name__ == "__main__":

    data_dir = "metadata"

    try:
        train_df, val_df, test_df = convert_to_seen_split_simple(
            data_dir=data_dir,
            split_index=1,
            train_ratio=0.7,
            val_ratio=0.2,
            seed=42
        )

        print("\nMinute!")

    except Exception as e:
        print(f"Error: {e}")
        print(":")
        print("1.  on Directory")
        print("2. metadataDirectory on datasplit1-train.csv, datasplit1-val.csv, datasplit1-test.csvFile")
        print("3. FileCPD_NAME（Name）")