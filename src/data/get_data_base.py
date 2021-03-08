import abc
import pandas as pd
from typing import Optional, Iterable, Tuple, Dict
import os
__all__ = ("data_base", )

class data_base(abc.ABC):

    data_dir :          Optional[str] = None
    raw_data_path :     Optional[str] = None
    interim_data_path : Optional[str] = None

    df :       Optional[pd.DataFrame] = None

    def _does_file_exist(self)-> bool:
        if os.path.exists(self.raw_data_path):
            print("Data path {} detected. Reading now...".format(self.raw_data_path))
            return True
        else:
            print("Data for MP not detected. Applying query now...")
            return False #self.get_data()

    def get_dataframe(self, sorted: Optional[bool] = True)-> pd.DataFrame:

        if self._does_file_exist():
            self.df = pd.read_pickle(self.raw_data_path)
        else:
            self.df = self._apply_query(sorted=sorted)
        print("Done")
        return(self.df)
