from dataset_specifics import DatasetSpecifics
from pydicom import dcmread

class VINDRSpecifics(DatasetSpecifics):
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
    def cc_col(self): return "CC"

    @property
    def mlo_col(self): return "MLO"
    
    @property
    def normalization_stats(self): return ([0.118, 0.118, 0.118], [0.1775, 0.1775, 0.1775])

    def load_img(self, path): return dcmread(path).pixel_array
