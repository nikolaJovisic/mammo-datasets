from dataset_specifics import *
import numpy as np

class EMBEDSpecifics(DatasetSpecifics):
    @property
    def name(self): return "embed"
    
    @property
    def label_col(self): return "asses"

    @property
    def view_col(self): return "ViewPosition"

    @property
    def laterality_col(self): return "ImageLateralityFinal"

    @property
    def study_col(self): return "acc_anon"

    @property
    def path_col(self): return "png_path"
    
    @property
    def normalization_stats(self): return ([0.118, 0.118, 0.118], [0.1775, 0.1775, 0.1775])

    @property
    def file_format(self): return FileFormat.PNG
    
    def raw_numpy(self, entry): return np.asarray(entry).astype(np.uint16)
    
    