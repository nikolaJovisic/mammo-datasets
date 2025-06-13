from dataset_specifics import DatasetSpecifics
from pydicom import dcmread

class CSAWSpecifics(DatasetSpecifics):
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
    def cc_col(self): return "CC"

    @property
    def mlo_col(self): return "MLO"
    
    @property
    def normalization_stats(self): return ([0.118, 0.118, 0.118], [0.1775, 0.1775, 0.1775])

    def load_img(self, path): return dcmread(path).pixel_array
