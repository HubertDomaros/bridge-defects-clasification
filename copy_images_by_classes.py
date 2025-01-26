#%%
import os
import shutil
import pandas as pd


#%%
def copy_images_by_classes(source_dir, out_dir, 
                           labels_json_path: str, img_col:str, labels_col: str):
    df = pd.read_json(labels_json_path)
    
    for class_id in df[labels_col].value_counts().index:
        print(class_id)
        os.makedirs(os.path.join(out_dir, str(class_id)), exist_ok=True)
        
    for index, data in df.iterrows():
        img = data[img_col]
        class_id = data[labels_col]
        
        print(os.path.abspath(os.path.join(source_dir, img)))

        shutil.copy2(os.path.abspath(os.path.join(source_dir, img)), 
                     os.path.abspath(os.path.join(out_dir, str(class_id), img)))
    
#%%
copy_images_by_classes('datasets/images/val', 'datasets/images/val',
                       'datasets/labels/val.json',
                       'img', 'combination_id')