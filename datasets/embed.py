from dataset_specifics import DatasetSpecifics
from PIL import Image

class EMBEDSpecifics(DatasetSpecifics):
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
    def cc_col(self): return "CC"

    @property
    def mlo_col(self): return "MLO"

    def load_img(self, path): return Image.open(path)
