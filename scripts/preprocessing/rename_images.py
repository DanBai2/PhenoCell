import os
import re
from pathlib import Path

def rename_batch_files(root_dir):
    """
    Directory，：
    1.  on Batch1-Plate1-20251019
    2. w1-w5Name
    """

    channel_mapping = {
        'w1': 'DAPI',
        'w2': 'FITC', 
        'w3': 'YFP',
        'w4': 'Texas Red',
        'w5': 'Cy5'
    }
    

    for file_path in Path(root_dir).rglob("*.tif"):
        filename = file_path.name
        

        # if filename.startswith("Batch1_"):
        #     new_filename = re.sub(r'^Batch1_', 'Batch1-Plate1-20251019_', filename)
        # else:
        #     new_filename = filename
        new_filename = filename
        


        for w_channel, channel_name in channel_mapping.items():

            pattern = f"({w_channel})([A-F0-9]{{8}}-[A-F0-9]{{4}}-[A-F0-9]{{4}}-[A-F0-9]{{4}}-[A-F0-9]{{12}})"
            match = re.search(pattern, new_filename, re.IGNORECASE)
            if match:

                new_filename = re.sub(
                    pattern, 
                    f"{channel_name}", 
                    new_filename
                )
        

        if new_filename != filename:
            new_file_path = file_path.parent / new_filename
            try:
                file_path.rename(new_file_path)
                print(f"Success: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Failed {filename}: {e}")

def preview_renames(root_dir):
    """
    ，
    """
    channel_mapping = {
        'w1': 'DAPI',
        'w2': 'FITC', 
        'w3': 'YFP',
        'w4': 'Texas Red',
        'w5': 'Cy5'
    }
    
    print(":")
    print("-" * 50)
    
    for file_path in Path(root_dir).rglob("*.tif"):
        filename = file_path.name
        

        if filename.startswith("Batch1_"):
            new_filename = re.sub(r'^Batch1_', 'Batch1-Plate1-20251019_', filename)
        else:
            new_filename = filename
        

        for w_channel, channel_name in channel_mapping.items():
            pattern = f"({w_channel})([A-F0-9]{{8}}-[A-F0-9]{{4}}-[A-F0-9]{{4}}-[A-F0-9]{{4}}-[A-F0-9]{{12}})"
            match = re.search(pattern, new_filename, re.IGNORECASE)
            if match:
                new_filename = re.sub(
                    pattern, 
                    f"{channel_name}{match.group(2)}", 
                    new_filename
                )
        
        if new_filename != filename:
            print(f"{filename} -> {new_filename}")

if __name__ == "__main__":


    root_directory='path/to/your/data/ours/img/Control-Model-Positive-Yao/Batch2/TimePoint_1'
    rename_batch_files(root_directory)
    # if not os.path.exists(root_directory):

    #     exit(1)
    

    # preview_renames(root_directory)
    


    
    # if confirm == 'y':
    #     rename_batch_files(root_directory)

    # else:

