import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

def extract_cell_line(site_id):
    """
    Extract cell type from site_id
    
    site_id format: HRCE-1_1_AA02_1 or HUVEC-1_1_AA02_1
    Extraction rule: get text before first underscore, then split by dash and take first part
    """
    if pd.isna(site_id):
        return None
    

    first_part = str(site_id).split('_')[0]
    


    cell_line = first_part.split('-')[0]
    
    return cell_line

def process_metadata_by_cell_line(metadata_path, output_dir='.'):
    """
    Process metadata.csv file, group by cell type
    
    Steps:
    1. CellType
    2. treatment，CellTypeGeneratecontrol.csv
    3. treatment，CellTypeMinuteMinute：
       a. unseenMinute: valtesttreatment on train
       b. seenMinute: valtesttreatment on train
    
    Args:
        metadata_path: metadata.csvFilePATH
        output_dir: Output directory
    """
    

    os.makedirs(output_dir, exist_ok=True)
    

    print(f"Reading file: {metadata_path}")
    df = pd.read_csv(metadata_path)
    

    required_columns = ['site_id', 'treatment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"CSVFile: {missing_columns}")
    
    print(f": {len(df)}")
    print(f": {df.columns.tolist()}")
    

    print(f"\nCellTypeInfo...")
    df['cell_line'] = df['site_id'].apply(extract_cell_line)
    

    cell_line_counts = df['cell_line'].value_counts()
    print(f"CellTypeMinute:")
    for cell_line, count in cell_line_counts.items():
        print(f"  {cell_line}: {count}  samples")
    

    control_mask = df['treatment'].isna() | (df['treatment'] == '')
    control_df = df[control_mask].copy()
    non_control_df = df[~control_mask].copy()
    
    print(f"\n1. ControlMinute:")
    print(f"   Control: {len(control_df)}")
    print(f"   Control: {len(non_control_df)}")
    

    print(f"\n2. ProcessingControl (CellType):")
    control_by_cell_line = control_df.groupby('cell_line')
    
    for cell_line, cell_control_df in control_by_cell_line:
        if len(cell_control_df) > 0:

            cell_control_df = cell_control_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            

            control_filename = f"{cell_line}-control.csv"
            control_path = os.path.join(output_dir, control_filename)
            cell_control_df.to_csv(control_path, index=False)
            print(f"   {cell_line}: {len(cell_control_df)}  samples -> {control_path}")
        else:
            print(f"   {cell_line}: Control")
    

    print(f"\n3. ProcessingControl (CellTypeMinutetrain/val/test):")
    

    non_control_by_cell_line = non_control_df.groupby('cell_line')
    

    unseen_results = {}
    seen_results = {}
    
    for cell_line, cell_df in non_control_by_cell_line:
        print(f"\n  ProcessingCellType: {cell_line}")
        print(f"    : {len(cell_df)}")
        

        unique_treatments = cell_df['treatment'].unique()
        print(f"    treatmentCount: {len(unique_treatments)}")
        

        treatment_counts = cell_df['treatment'].value_counts()
        print(f"    treatment: {treatment_counts.mean():.2f}")
        

        print(f"\n    A. UNSEENMinute (valtesttreatment on train):")
        

        np.random.seed(42)
        

        if len(unique_treatments) < 3:
            print(f"      Warning: {cell_line}treatmentCount3，unseenMinute")
        else:

            shuffled_treatments = np.random.permutation(unique_treatments)
            

            n_treatments = len(shuffled_treatments)
            train_end = int(n_treatments * 0.7)
            val_end = train_end + int(n_treatments * 0.2)
            

            train_treatments = shuffled_treatments[:train_end]
            val_treatments = shuffled_treatments[train_end:val_end]
            test_treatments = shuffled_treatments[val_end:]
            
            print(f"      Train treatmentsCount: {len(train_treatments)}")
            print(f"      Val treatmentsCount: {len(val_treatments)}")
            print(f"      Test treatmentsCount: {len(test_treatments)}")
            

            train_df = cell_df[cell_df['treatment'].isin(train_treatments)].copy()
            val_df = cell_df[cell_df['treatment'].isin(val_treatments)].copy()
            test_df = cell_df[cell_df['treatment'].isin(test_treatments)].copy()
            
            print(f"      Train: {len(train_df)} ({len(train_df)/len(cell_df)*100:.1f}%)")
            print(f"      Val: {len(val_df)} ({len(val_df)/len(cell_df)*100:.1f}%)")
            print(f"      Test: {len(test_df)} ({len(test_df)/len(cell_df)*100:.1f}%)")
            

            train_treatments_set = set(train_treatments)
            val_treatments_set = set(val_treatments)
            test_treatments_set = set(test_treatments)
            
            overlap_train_val = len(train_treatments_set.intersection(val_treatments_set))
            overlap_train_test = len(train_treatments_set.intersection(test_treatments_set))
            overlap_val_test = len(val_treatments_set.intersection(test_treatments_set))
            
            if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
                print(f"      Warning: {cell_line}unseenMinute on treatment!")
            else:
                print(f"      Validation: {cell_line}unseenMinuteDatasettreatment")
            

            train_df = train_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            val_df = val_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            test_df = test_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            

            train_filename = f"{cell_line}-unseen-train.csv"
            val_filename = f"{cell_line}-unseen-val.csv"
            test_filename = f"{cell_line}-unseen-test.csv"
            
            train_path = os.path.join(output_dir, train_filename)
            val_path = os.path.join(output_dir, val_filename)
            test_path = os.path.join(output_dir, test_filename)
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"      FileSave:")
            print(f"        {train_path}")
            print(f"        {val_path}")
            print(f"        {test_path}")
            

            unseen_results[cell_line] = {
                'train': train_df,
                'val': val_df,
                'test': test_df,
                'train_treatments': train_treatments,
                'val_treatments': val_treatments,
                'test_treatments': test_treatments
            }
        

        print(f"\n    B. SEENMinute (valtesttreatment on train):")
        

        np.random.seed(42)
        

        if len(cell_df) < 10:
            print(f"      Warning: {cell_line}10，seenMinute")
        else:



            

            shuffled_indices = np.random.permutation(cell_df.index)
            

            n_samples = len(cell_df)
            train_end = int(n_samples * 0.7)
            val_end = train_end + int(n_samples * 0.2)
            

            train_indices = shuffled_indices[:train_end]
            val_indices = shuffled_indices[train_end:val_end]
            test_indices = shuffled_indices[val_end:]
            

            train_df = cell_df.loc[train_indices].copy()
            val_df = cell_df.loc[val_indices].copy()
            test_df = cell_df.loc[test_indices].copy()
            
            print(f"      Train: {len(train_df)} ({len(train_df)/len(cell_df)*100:.1f}%)")
            print(f"      Val: {len(val_df)} ({len(val_df)/len(cell_df)*100:.1f}%)")
            print(f"      Test: {len(test_df)} ({len(test_df)/len(cell_df)*100:.1f}%)")
            

            train_treatments_set = set(train_df['treatment'].unique())
            val_treatments_set = set(val_df['treatment'].unique())
            test_treatments_set = set(test_df['treatment'].unique())
            

            overlap_train_val = len(train_treatments_set.intersection(val_treatments_set))
            overlap_train_test = len(train_treatments_set.intersection(test_treatments_set))
            overlap_val_test = len(val_treatments_set.intersection(test_treatments_set))
            
            print(f"      TrainVal treatment: {overlap_train_val}")
            print(f"      TrainTest treatment: {overlap_train_test}")
            print(f"      ValTest treatment: {overlap_val_test}")
            

            all_treatments = set(cell_df['treatment'].unique())
            treatments_in_all_sets = 0
            treatments_in_train_val = 0
            treatments_in_train_test = 0
            
            for treatment in all_treatments:
                in_train = treatment in train_treatments_set
                in_val = treatment in val_treatments_set
                in_test = treatment in test_treatments_set
                
                if in_train and in_val and in_test:
                    treatments_in_all_sets += 1
                elif in_train and in_val:
                    treatments_in_train_val += 1
                elif in_train and in_test:
                    treatments_in_train_test += 1
            
            print(f"       on treatmentCount: {treatments_in_all_sets}")
            print(f"       on trainvaltreatmentCount: {treatments_in_train_val}")
            print(f"       on traintesttreatmentCount: {treatments_in_train_test}")
            

            train_df = train_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            val_df = val_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            test_df = test_df.rename(columns={'site_id': 'SAMPLE_KEY'})
            

            train_filename = f"{cell_line}-seen-train.csv"
            val_filename = f"{cell_line}-seen-val.csv"
            test_filename = f"{cell_line}-seen-test.csv"
            
            train_path = os.path.join(output_dir, train_filename)
            val_path = os.path.join(output_dir, val_filename)
            test_path = os.path.join(output_dir, test_filename)
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"      FileSave:")
            print(f"        {train_path}")
            print(f"        {val_path}")
            print(f"        {test_path}")
            

            seen_results[cell_line] = {
                'train': train_df,
                'val': val_df,
                'test': test_df,
                'train_treatments': train_treatments_set,
                'val_treatments': val_treatments_set,
                'test_treatments': test_treatments_set
            }
    

    print(f"\n4. Generate...")
    

    unseen_stats_path = os.path.join(output_dir, 'unseen_data_split_stats.txt')
    with open(unseen_stats_path, 'w') as f:
        f.write("UNSEENMinute (CellType)\n")
        f.write("="*70 + "\n")
        f.write(f": {len(df)}\n")
        f.write(f"Control: {len(control_df)}\n")
        f.write(f"Control: {len(non_control_df)}\n")
        f.write(f"CellType: {len(cell_line_counts)}\n\n")
        
        f.write("CellTypeUNSEENMinute:\n")
        for cell_line in cell_line_counts.index:
            f.write(f"\n{cell_line}:\n")
            f.write(f"  : {cell_line_counts[cell_line]}\n")
            

            cell_control_count = len(control_df[control_df['cell_line'] == cell_line]) if len(control_df) > 0 else 0
            f.write(f"  Control: {cell_control_count}\n")
            

            cell_non_control_count = len(non_control_df[non_control_df['cell_line'] == cell_line]) if len(non_control_df) > 0 else 0
            f.write(f"  Control: {cell_non_control_count}\n")
            
            if cell_line in unseen_results:
                train_count = len(unseen_results[cell_line]['train'])
                val_count = len(unseen_results[cell_line]['val'])
                test_count = len(unseen_results[cell_line]['test'])
                total = train_count + val_count + test_count
                
                if total > 0:
                    f.write(f"  UNSEENMinuteResults:\n")
                    f.write(f"    Train: {train_count} ({train_count/total*100:.1f}%)\n")
                    f.write(f"    Val: {val_count} ({val_count/total*100:.1f}%)\n")
                    f.write(f"    Test: {test_count} ({test_count/total*100:.1f}%)\n")
                    f.write(f"    Train treatmentsCount: {len(unseen_results[cell_line]['train_treatments'])}\n")
                    f.write(f"    Val treatmentsCount: {len(unseen_results[cell_line]['val_treatments'])}\n")
                    f.write(f"    Test treatmentsCount: {len(unseen_results[cell_line]['test_treatments'])}\n")
            else:
                f.write(f"  UNSEENMinute: Minute (treatmentCount)\n")
    
    print(f"   UNSEENSave: {unseen_stats_path}")
    

    seen_stats_path = os.path.join(output_dir, 'seen_data_split_stats.txt')
    with open(seen_stats_path, 'w') as f:
        f.write("SEENMinute (CellType)\n")
        f.write("="*70 + "\n")
        f.write(f": {len(df)}\n")
        f.write(f"Control: {len(control_df)}\n")
        f.write(f"Control: {len(non_control_df)}\n")
        f.write(f"CellType: {len(cell_line_counts)}\n\n")
        
        f.write("CellTypeSEENMinute:\n")
        for cell_line in cell_line_counts.index:
            f.write(f"\n{cell_line}:\n")
            f.write(f"  : {cell_line_counts[cell_line]}\n")
            

            cell_control_count = len(control_df[control_df['cell_line'] == cell_line]) if len(control_df) > 0 else 0
            f.write(f"  Control: {cell_control_count}\n")
            

            cell_non_control_count = len(non_control_df[non_control_df['cell_line'] == cell_line]) if len(non_control_df) > 0 else 0
            f.write(f"  Control: {cell_non_control_count}\n")
            
            if cell_line in seen_results:
                train_count = len(seen_results[cell_line]['train'])
                val_count = len(seen_results[cell_line]['val'])
                test_count = len(seen_results[cell_line]['test'])
                total = train_count + val_count + test_count
                
                if total > 0:
                    f.write(f"  SEENMinuteResults:\n")
                    f.write(f"    Train: {train_count} ({train_count/total*100:.1f}%)\n")
                    f.write(f"    Val: {val_count} ({val_count/total*100:.1f}%)\n")
                    f.write(f"    Test: {test_count} ({test_count/total*100:.1f}%)\n")
                    f.write(f"    Train treatmentsCount: {len(seen_results[cell_line]['train_treatments'])}\n")
                    f.write(f"    Val treatmentsCount: {len(seen_results[cell_line]['val_treatments'])}\n")
                    f.write(f"    Test treatmentsCount: {len(seen_results[cell_line]['test_treatments'])}\n")
                    

                    train_treatments_set = seen_results[cell_line]['train_treatments']
                    val_treatments_set = seen_results[cell_line]['val_treatments']
                    test_treatments_set = seen_results[cell_line]['test_treatments']
                    
                    overlap_train_val = len(train_treatments_set.intersection(val_treatments_set))
                    overlap_train_test = len(train_treatments_set.intersection(test_treatments_set))
                    overlap_val_test = len(val_treatments_set.intersection(test_treatments_set))
                    
                    f.write(f"    Treatment:\n")
                    f.write(f"      TrainVal: {overlap_train_val}\n")
                    f.write(f"      TrainTest: {overlap_train_test}\n")
                    f.write(f"      ValTest: {overlap_val_test}\n")
            else:
                f.write(f"  SEENMinute: Minute ()\n")
    
    print(f"   SEENSave: {seen_stats_path}")
    
    return unseen_results, seen_results


def main():

    metadata_path = 'path/to/your/data/RXRX19a/RxRx19a/metadata.csv'
    output_dir = 'path/to/your/data/RXRX19a/RxRx19a'
    

    try:
        print("StartProcessingmetadata.csvFile...")
        print("="*60)
        
        unseen_results, seen_results = process_metadata_by_cell_line(metadata_path, output_dir)
        
        print("\n" + "="*60)
        print("Processing!")
        print(f"FileSave: {output_dir}/")
        






        
    except Exception as e:
        print(f"ProcessingError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()