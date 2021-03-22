import pandas as pd
import logging
import sys
from pymatgen.symmetry.groups import SYMM_DATA, sg_symbol_from_int_number


def sortByMPID(df: pd.DataFrame) -> pd.DataFrame:
    mpid_num = []
    for i in df["material_id"]:
        mpid_num.append(int(i[3:]))
    df["mpid_num"] = mpid_num
    df = df.sort_values(by="mpid_num").reset_index(drop=True)
    df = df.drop(columns=["mpid_num"])
    return df

def filterIDs(df: pd.DataFrame) -> pd.DataFrame:
    unsupportedMPIDs = ["mp-28709",  #C120S32
                        "mp-28905",  #Sr6C120
                        "mp-28979",  #Ba6C120
                        "mp-29281",  #Th24P132
                        "mp-555563", #PH6C2S2NCl2O4 #DOI: 10.17188/1268877
                        "mp-560718", #Te4H48Au4C16S12N4
                        "mp-568028", #C120
                        "mp-568259", #Ta4Si8P4H72C24N8Cl24
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

def polarGroupUsedInMP():
    """
    Materials Project has more space groups than normal convention. This function finds
    all the polar groups for materials project extended list of space groups.
    """
    # This is a list of the point groups as noted in pymatgen
    point_groups = []
    for i in range(1,231):
        symbol = sg_symbol_from_int_number(i)
        point_groups.append(SYMM_DATA['space_group_encoding'][symbol]['point_group'])

    # Note that there are 40 of them, rather than 32.
    print("Number of point groups denoted in pymatgen: {}".format(len(set(point_groups))))

    # This is because multiple conventions are used for the same point group.
    # This dictionary can be used to convert between them.
    point_group_conv = {'321' :'32', '312': '32', '3m1' :'3m', '31m': '3m',
                        '-3m1' : '-3m', '-31m': '-3m', '-4m2': '-42m', '-62m': '-6m2' }

    # Using this dictionary we can correct to the standard point group notation.
    corrected_point_groups = [point_group_conv.get(pg, pg) for pg in point_groups]
    # Which produces the correct number of point groups. 32.
    print("Number of point groups in conventional notation: {}".format(len(set(corrected_point_groups))))

    # There are 10 polar point groups
    polar_point_groups = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']

    # Polar spacegroups have polar point groups.
    polar_spacegroups = []

    # There are 230 spacegroups
    for i in range(1,231):
        symbol = sg_symbol_from_int_number(i)
        pg = SYMM_DATA['space_group_encoding'][symbol]['point_group']
        if point_group_conv.get(pg, pg) in polar_point_groups:
            polar_spacegroups.append(i)

    # 68 of the 230 spacegroups are polar.
    print("Number of polar spacegroups: {}".format(len(polar_spacegroups)))

    return polar_spacegroups

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)
