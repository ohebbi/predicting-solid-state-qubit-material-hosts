import pandas as pd
import logging
import sys
def sortByMPID(df):
    mpid_num = []
    for i in df["material_id"]:
        mpid_num.append(int(i[3:]))
    df["mpid_num"] = mpid_num
    df = df.sort_values(by="mpid_num").reset_index(drop=True)
    df = df.drop(columns=["mpid_num"])
    return df

def filterIDs(df):
    unsupportedMPIDs = ["mp-28709",  #C120S32
                        "mp-28905",  #Sr6C120
                        "mp-28979",  #Ba6C120
                        "mp-29281",  #Th24P132
                        "mp-555563", #PH6C2S2NCl2O4 #DOI: 10.17188/1268877
                        "mp-583476", #Nb7S2I19      #DOI: 10.17188/1277059
                        "mp-600205", #H10C5SeS2N3Cl          #DOI: -
                        "mp-600217", #H80C40Se8S16Br8N24     #DOI: -
                        "mp-1195290", #Ga3Si5P10H36C12N4Cl11 #DOI: -
                        "mp-1196358", #P4H120Pt8C40I8N4Cl8   #DOI: -
                        "mp-1196439", #Sn8P4H128C44N12Cl8O4  #DOI: -
                        "mp-1198652", #Te4H72C36S24N12Cl4    #DOI: -
                        "mp-1198926", #Re8H96C24S24N48Cl48   #DOI: -
                        "mp-1199490", #Mn4H64C16S16N32Cl8    #DOI: -
                        "mp-1199686", #Mo4P16H152C52N16Cl16  #DOI: -
                        "mp-1203403", #C121S2Cl20            #DOI: -
                        "mp-1204279", #Si16Te8H176Pd8C64Cl16 #DOI: -
                        "mp-1204629"] #P16H216C80N32Cl8     #DOI: -
    print("A total of {} MPIDs are inconsistent with the rest."
          .format(len(unsupportedMPIDs)))

    for unsupportedMPID in unsupportedMPIDs:
        if unsupportedMPID in list(df["material_id"].values):
            df = df.drop(df[df["material_id"] == str(unsupportedMPID)].index)

    df = df.reset_index(drop=True)
    return df

def countSimilarEntriesWithMP(listOfEntries, nameOfDatabase):
    similarEntries = 0
    for i in listOfEntries:
        if i>=0:
            similarEntries += 1

    LOG.info("The amount of similar entries between MP and {} is {},".format(nameOfDatabase, similarEntries))
    LOG.info("which is {} percent".format(similarEntries/len(listOfEntries)))

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)
