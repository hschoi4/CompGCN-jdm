#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import dataclasses
from typing import Set, Any
import xml.etree.ElementTree as ET

def get_entities(df_nodes, export):
    """
        Get label of each node and new ids
    """
    entities = df_nodes['lexname'].tolist()
    list_labels = []

    # Get the label and features between tags in xml format
    for ent in entities:
        tag = ET.fromstring('<tag>'+ent+"</tag>")
        acc = ""
        for field in tag:
            acc = acc + " " + field.text
        list_labels.append(acc.lstrip())

    df_nodes['label'] = list_labels

    if export:
        df_nodes['label'].to_csv('data/RLF/entities.txt', index=True, sep='\t', header=False)

    return df_nodes

def get_relations(df_lffam, df_cp, export):
    """
        Get relation types and new ids
    """
    df_relations = pd.concat([df_lffam, df_cp], ignore_index=True)

    if export:
        relations['name'].to_csv('data/RLF/relations.txt', index=True, sep="\t", header=False)

    return df_relations

def normalize_label(df, entities, relations, export):
    """
        Put new ids for relations and entities
    """

    ent_map = {id_node : i for i, id_node in enumerate(entities['id'].tolist())}
    rel_map = {id_rel : i for i, id_rel in enumerate(relations['id'].tolist())}

    df['t'] = df['t'].map(rel_map)
    df['n1'] = df['n1'].map(ent_map)
    df['n2'] = df['n2'].map(ent_map)

    df = df.drop(columns='id')

    if export:
        df.to_csv('data/RLF/all_triplets.txt', index=False, sep="\t", header=True)

    return df

def get_default_format(df, entities, relations, export):
    """
        Put labels for relations and entities
    """

    ent_map = dict(zip(entities['id'].tolist(), entities['label'].tolist()))
    print(ent_map)
    rel_map = dict(zip(relations['id'].tolist(), relations['name'].tolist()))

    df['t'] = df['t'].map(rel_map)
    df['n1'] = df['n1'].map(ent_map)
    df['n2'] = df['n2'].map(ent_map)

    df = df.drop(columns='id')

    if export:
        df.to_csv('data/RLF/all_triplets_with_labels.txt', index=False, sep="\t", header=True)

    return df


if __name__ == '__main__':

    df_nodes = pd.read_csv("data/RLF/originals/01-lsnodes.csv", sep="\t")
    df_lffam = pd.read_csv("data/RLF/originals/lffam_ids_names.csv", sep="\t")      # lexical function families edges
    df_cp = pd.read_csv("data/RLF/originals/cp_ids_names.csv", sep="\t")            # copolysemy edges

    entities = get_entities(df_nodes, True)
    relations = get_relations(df_lffam, df_cp, False)

    df_triplets = pd.read_csv("data/RLF/originals/rlf_lffam_cp.csv", sep="\t", index_col=False)
    # df_triplets_normalized = normalize_label(df_triplets, entities, relations, False)
    df = get_default_format(df_triplets, entities, relations, True)
