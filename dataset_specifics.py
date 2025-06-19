from abc import ABC, abstractmethod
from enum import Enum, auto
from PIL import Image
from pydicom import dcmread

class FileFormat(Enum):
    DICOM = auto()
    PNG = auto()
    
_READ_FILE = {
    FileFormat.DICOM: lambda path: dcmread(path),
    FileFormat.PNG: lambda path: Image.open(path),
}

class DatasetSpecifics(ABC):
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def label_col(self):
        pass

    @property
    @abstractmethod
    def view_col(self):
        pass

    @property
    @abstractmethod
    def laterality_col(self):
        pass

    @property
    @abstractmethod
    def study_col(self):
        pass

    @property
    @abstractmethod
    def path_col(self):
        pass
    
    @property
    @abstractmethod
    def normalization_stats(self):
        pass
    
    @property
    def cc_col(self):
        return "CC"

    @property
    def mlo_col(self):
        return "MLO"
    
    @property
    def file_format(self): return FileFormat.DICOM
    
    def raw_numpy(self, entry): return entry.pixel_array 

    def read_file(self, path):
        return _READ_FILE[self.file_format](path)
    
    def get_agg_columns(self):
        return (
            self.label_col,
            self.view_col,
            self.laterality_col,
            self.study_col,
            self.path_col,
            self.cc_col,
            self.mlo_col,
        )
    