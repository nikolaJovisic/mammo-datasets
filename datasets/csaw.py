from dataset_specifics import *

class CSAWSpecifics(DatasetSpecifics):
    @property
    def name(self): return "csaw"
    
    @property
    def label_col(self): return "rad_recall"

    @property
    def view_col(self): return "viewposition"

    @property
    def laterality_col(self): return "imagelaterality"

    @property
    def study_col(self): return "anon_patientid"

    @property
    def path_col(self): return "anon_filename"
    
    @property
    def normalization_stats(self): raise NotImplementedError
