# -*- coding: utf-8 -*-
from typing import Iterable, Optional
import os
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from . import utils
from . import get_data_base

class data_Citrine(get_data_base.data_base):
    def __init__(self, API_KEY: str):

        self.API_KEY = API_KEY
        self.data_dir = Path.cwd().parent / "data"
        self.raw_data_path = self.data_dir/ "raw" / "Citrine" / "Citrine.pkl"
        self.interim_data_path = self.data_dir / "interim" / "Citrine" / "Citrine.pkl"
        self.df = None

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:
        cdr = CitrineDataRetrieval(api_key=self.API_KEY)
        criteria  = {"data_type": "EXPERIMENTAL"}
        properties = ['Band gap']
        common_fields = ["uid","chemicalFormula", "references", "Crystallinity", "Structure", "Crystal structure", "uid"]

        self.df = cdr.get_dataframe(criteria = criteria,
                           properties = properties,
                           common_fields = common_fields)

        print("Writing to raw data...")
        self.df.to_pickle(self.raw_data_path)
        return self.df;

    def _sort(self, entries: pd.DataFrame)-> pd.DataFrame:

        self.df = self.df[self.df["Band gap-dataType"]=="EXPERIMENTAL"]\
                                            .dropna(axis=1, how='all')

        bandgap = np.empty(len(entries["full_formula"]))
        bandgap[:] = np.nan

        for i, entry in tqdm(enumerate(entries["full_formula"])):
            for j, exp in enumerate(self.df["chemicalFormula"]):
                if entry == exp and float(self.df["Band gap"].iloc[j])>=0:
                    bandgap[i] = float(self.df["Band gap"].iloc[j])
        sorted_df = pd.DataFrame({"citrine_bg": bandgap})

        return sorted_df

    def sort_with_MP(self, entries: pd.DataFrame)-> np.array:
        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(entries)
        utils.countSimilarEntriesWithMP(sorted_df["citrine_bg"], "Citrine")
        return sorted_df
