from dataset_specifics import *

class RSNASpecifics(DatasetSpecifics):
    @property
    def name(self): return "rsna"
    
    @property
    def label_col(self): return "label"

    @property
    def view_col(self): return "view"

    @property
    def laterality_col(self): return "laterality"

    @property
    def study_col(self): return "patient_id"

    @property
    def path_col(self): return "dcm_path"

    @property
    def normalization_stats(self): raise NotImplementedError