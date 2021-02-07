from matminer.featurizers import composition as cf
from pymatgen import Composition
from tqdm import tqdm
# pandas
import pandas as pd
import numpy as np
# Load featurizers and conversion functions
from matminer.featurizers.conversions import StrToComposition,CompositionToOxidComposition
from matminer.featurizers.base import MultipleFeaturizer
from helperFunctions.keepTypesReadWrite import read_csv, to_csv

import time

def applyOxidFeaturizers(entries, fileName):

    designMatrix = StrToComposition().featurize_dataframe(entries[["full_formula", "pretty_formula", "material_id"]], "pretty_formula")
    print(len(designMatrix.columns))

    designMatrix = cf.ElementFraction().featurize_dataframe(designMatrix, "composition")
    print(len(designMatrix.columns))

    tempDf = pd.DataFrame({})
    last_iteration=0


    for rows in tqdm(range(0,len(entries),1)):
        print(designMatrix[["full_formula","pretty_formula"]].iloc[last_iteration:rows])

        portion = CompositionToOxidComposition().featurize_dataframe(designMatrix.iloc[last_iteration:rows], "composition",ignore_errors=True)
        print(designMatrix[["full_formula","pretty_formula","composition"]].iloc[last_iteration:rows])
        portion = cf.ElectronegativityDiff().featurize_dataframe(portion, "composition_oxid",ignore_errors=True)

        tempDf = pd.concat([tempDf,portion]).reset_index(drop=True)
        last_iteration = rows

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)

        if (fileName) and (rows % 50 == 0):
            tempDf.to_csv(fileName, sep=",", index = False)

    portion = CompositionToOxidComposition().featurize_dataframe(designMatrix.iloc[last_iteration:len(entries)], "composition",ignore_errors=True)
    portion = cf.ElectronegativityDiff().featurize_dataframe(portion, "composition_oxid",ignore_errors=True)
    tempDf = pd.concat([tempDf,portion]).reset_index(drop=True)
    print(tempDf)
    tempDf.to_csv(fileName, sep=",", index = False)

def queueOxidFeaturizers():
    directory = "data/databases/initialDataMP"

    entries = pd.read_csv(directory + "/MP/MP_FLAGBIGFILE.csv", sep=",")
    print(entries.columns)

    fileName = "data/databases/initialDataMP/MP/MP_oxidationFeaturized.csv"

    applyOxidFeaturizers(entries, fileName)


def reQueueOxidFeaturizers():

    directory = "../dataMining/data/databases/initialDataMP"

    #reading entries from MP
    MP_entries = pd.read_csv(directory+"/MP/MP_FLAGBIGFILE.csv", sep=",")
    print(MP_entries.iloc[24671])
    MP_entries = MP_entries.drop([24671]) # avoid index

    #print(MP_entries[MP_entries["full_formula"]=="Ta8Ag4O22"])

    previous_MP_featurizer = pd.read_csv(directory+"/MP/MP_oxidationFeaturized.csv", sep=",")

    howFar = MP_entries[MP_entries["material_id"] == previous_MP_featurizer["material_id"].iloc[-1]].index.values
    print(howFar)
    print(previous_MP_featurizer.iloc[howFar])
    print(MP_entries.iloc[howFar])

    #previous_MP_featurizer.to_csv(directory+"/MP/MP_oxidationFeaturized_" + str(howFar[0]) + ".csv", sep=",", index=False)

    #applyOxidFeaturizers(entries=MP_entries.iloc[howFar[0]+1:], fileName=directory+"/MP/MP_oxidationFeaturized_REQUEUE.csv")

    return;
if __name__ == '__main__':
    #queueOxidFeaturizers()
    reQueueOxidFeaturizers()
