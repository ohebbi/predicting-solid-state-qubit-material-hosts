import qmpy_rester as qr
import pandas as pd
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval

mdf = MDFDataRetrieval (anonymous = True)

directory = "../dataMining/data/databases/OQMD/"

OQMDentries = mdf.get_dataframe({
                "source_names": ['oqmd']#,
                #"match_fields":{"oqmd.converged": True},
                },
                unwind_arrays=False)
OQMDentries.to_csv(directory+"/OQMD_FLAGBIGFILE.csv", sep=",", index = False)
