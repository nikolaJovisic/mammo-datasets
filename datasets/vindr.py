from dataset_specifics import *

class VINDRSpecifics(DatasetSpecifics):
    @property
    def name(self): return "vindr"
    
    @property
    def label_col(self): return "breast_birads"

    @property
    def view_col(self): return "view_position"

    @property
    def laterality_col(self): return "laterality"

    @property
    def study_col(self): return "study_id"

    @property
    def path_col(self): return "dcm_path"
    
    @property
    def normalization_stats(self): raise NotImplementedError
        
    def map_label(self, label):
        birads = int(label[-1]) 
        if birads == 1:
            return 0
        if birads in [4, 5]:
            return 1
        if birads in [2, 3]:
            return 2
        raise ValueError('Unknown birads in vindr.')