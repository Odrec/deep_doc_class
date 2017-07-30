# coding=utf-8

import os, sys
from os.path import join, realpath, dirname, isdir, basename, isfile
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)

import pandas as pd
import json

from multiprocessing import Pool

from doc_globals import*

FEATURES_NAMES = ["filename","folder_name"]

def load_single_metarow(doc_id, fields, metadata):
    metadict = {}
    if(type(metadata)==str and isfile(metadata)):
        metadata=pd.read_csv(path, delimiter=',', quoting=1, encoding='utf-8')
        pd_series = metadata[metadata['document_id'] == doc_id]
        if(pd_series.empty):
            print("No csv metadata for doc id %s!!!" %(doc_id,))
            for field in fields:
                metadict[field] = None
        else:
            for field in fields:
                metadict[field] = metadata[field]
    else:
        for field in fields:
            metadict[field] = metadata[field]
    return metadict

def load_single_metafield(doc_ids, field, metadata=join(DATA_PATH,"classified_metadata.csv")):
    if(type(metadata)==str and isfile(metadata)):
        metadata=pd.read_csv(metadata, delimiter=',', quoting=1, encoding='utf-8')
    if(type(doc_ids)==str):
        selected_data = metadata[metadata["document_id"]==doc_ids]
        if(selected_data.empty):
            print("No csv metadata for doc id %s!!!" %(doc_ids,))
            sys.exit(1)
        else:
            selected_data = selected_data[field].values[0]
    else:
        selected_data = list(metadata[metadata["document_id"].isin(doc_ids)][field])
        if(len(selected_data)<len(doc_ids)):
            selected_data = []
            metadata = metadata.set_index("document_id")
            for d_id in doc_ids:
                if(d_id in metadata.index):
                    selected_data.append(metadata.loc[d_id][field])
                else:
                    selected_data.append("")
            # print("No csv metadata for %d doc_ids!!!" %(len(doc_ids)-len(selected_data),))
            # sys.exit(1)

    return selected_data

if __name__ == "__main__":
    doc_ids = "76ae7c120910a7830a1c0e0262d8cc5e"
    doc_ids2 = ['76ae7c120910a7830a1c0e0262d8cc5e', '76b43039b7577b89dcad89621d42c7d5']

    res = load_single_metafield(doc_ids, field="folder_name")
    print(res)
    res = load_single_metafield(doc_ids2, field="folder_name")
    print(res)
