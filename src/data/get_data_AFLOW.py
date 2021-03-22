# -*- coding: utf-8 -*-
from typing import Optional, Iterable, Dict
import os
import pandas as pd
import numpy as np
import wget
import pickle

from pathlib import Path
from tqdm import tqdm
from src.data.utils import countSimilarEntriesWithMP, LOG
from aflow import *

from src.data.get_data_MP import data_MP
from src.data import get_data_base


class data_AFLOW(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None, MAPI_KEY: Optional[str] = None):

        self.API_KEY = API_KEY
        self.MAPI_KEY = API_KEY
        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.raw_data_path = self.data_dir / "raw" / "AFLOW" / "AFLOW.pkl"
        self.interim_data_path = self.data_dir / "interim" / "AFLOW" / "AFLOW.pkl"
        self.df = None
        super().__init__()

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Add unique url id for figshare endpoint
        url = "https://ndownloader.figshare.com/files/26777717"
        file = wget.download(url)

        # Read and load pkl data
        with open(file, 'rb') as f:
            self.df = pickle.load(f)
            os.remove(file)

        # TODO : Add option to make new queries to AFLOWML
        """
        try:
            MP = data_MP(API_KEY = self.MAPI_KEY)
        except:
            raise ValueError("AFLOW is dependent on MP data. Add MAPI_KEY argument\
            to class constructor.")

        entries = MP.get_dataframe()

        compound_list = list(entries["full_formula"])
        #choosing keys used in AFLOW. We will here use all features in AFLOW.
        keys = list(pd.read_pickle(Path.cwd().parent / "data" / "raw" / "AFLOW" / "AFLOW_keywords.pkl").columns)


        self.df = get_dataframe_AFLOW(compound_list=compound_list, keys=keys, batch_size = 1000, catalog="icsd")
        """

        LOG.info("Writing to raw data...")
        self.df.to_pickle(self.data_dir / "raw"  / "AFLOW" / "AFLOW.pkl")

        return self.df;

    def get_data_AFLOW(self, compound_list: list, keys: list, batch_size: int, catalog: str = "icsd")-> Dict :
        """
        A function used to make a query to AFLOW.
        ...
        Args
        ----------
        compound_list : list (dim:N)
            A list of strings containing full formula, eg. H2O1 or Si1C1
        keys : list (dim:M)
            A list containing the features of the compound, found in documentation of AFLUX.
            eg. Egap
        batch_size : int
            Number of data entries to return per HTTP request
        catalog : str
            "icsd" for ICSD

        Returns
        -------
        dict
            A dictionary containing the resulting matching queries. This can result
            in several matching compounds for each compound.
        """
        index = 0
        aflow_dict = {k: [] for k in keys}
        for compound in tqdm(compound_list):
            LOG.info("Current query: {}".format(compound))

            results = search(catalog=catalog, batch_size=batch_size)\
                .filter(K.compound==compound)

            if len(results)>0:
                for result in tqdm(results):
                    for key in keys:
                        try:
                            aflow_dict[key].append(getattr(result,key))
                        except:
                            aflow_dict[key].append("None")
                    if (index % 10 == 0):
                        pd.DataFrame.from_dict(aflow_dict).to_pickle(self.data_dir / "raw"  / "AFLOW" / "new_AFLOW.pkl")

                    index += 1
            else:
                LOG.info("No compound is matching the search")
                continue

        return aflow_dict

    def get_dataframe_AFLOW(self, compound_list: list, keys: list, batch_size: int, catalog: str = "icsd")-> pd.DataFrame:
        """
        A function used to make a query to AFLOW.
        ...
        Args
        ----------
        See get_Data_AFLOW()

        Returns
        -------
        pd.DataFrame (dim:MxN)
            A DataFrame containing the resulting matching queries. This can result
            in several matching compounds for each compound.
        """
        return pd.DataFrame.from_dict(get_data_AFLOW(compound_list, keys, batch_size, catalog, fileName))

    def _sort(self, entries: pd.DataFrame)-> pd.DataFrame:

        bandgap    = np.empty(len(entries))
        bandgap[:] = np.nan

        bandgap_fitted    = np.copy(bandgap)
        spacegroup_orig   = np.copy(bandgap)
        spacegroup_relax  = np.copy(bandgap)
        ICSDs             = np.copy(bandgap)

        LOG.info("total iterations: {}".format(len(entries)))
        for i, icsd_list in tqdm(enumerate(entries["icsd_ids"])):
            for j, aflow_icsd in enumerate(self.df["prototype"]):
                for icsd in eval(str(icsd_list)):
                    if icsd == int(aflow_icsd.split("_")[-1][:-1]):

                        spacegroup_orig[i]  = int(self.df["spacegroup_orig"] .iloc[j])
                        spacegroup_relax[i] = int(self.df["spacegroup_relax"].iloc[j])
                        ICSDs[i]             = int(aflow_icsd.split("_")[-1][:-1])
                        bandgap[i]        = self.df["Egap"]     .iloc[j]
                        bandgap_fitted[i] = self.df["Egap_fit"] .iloc[j]

        sorted_df = pd.DataFrame({"aflow_bg":     bandgap,
                                 "aflow_bg_fit":  bandgap_fitted,
                                 "aflow_sg_orig": spacegroup_orig,
                                 "aflow_sg_relax":spacegroup_relax,
                                 "aflow_icsd":    ICSDs})

        sorted_df.to_pickle(self.data_dir / "interim" / "AFLOW" / "AFLOW.pkl")
        return sorted_df

    def sort_with_MP(self, entries: pd.DataFrame)-> pd.DataFrame:

        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(entries)
        countSimilarEntriesWithMP(sorted_df["aflow_bg"], "AFLOW")
        countSimilarEntriesWithMP(sorted_df["aflow_bg_fit"], "AFLOW Fit")
        return sorted_df
