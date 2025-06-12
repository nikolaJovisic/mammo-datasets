from abc import ABC, abstractmethod

class DatasetSpecifics(ABC):
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
    def cc_col(self):
        pass

    @property
    @abstractmethod
    def mlo_col(self):
        pass
    
    @abstractmethod
    def load_img(self, path):
        pass
    
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
    