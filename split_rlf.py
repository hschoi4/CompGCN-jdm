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

RLF_PATH = "data/RLF/"

@dataclasses.dataclass
class Triplets:
    train: Any
    valid: Any
    test: Any

    def export(self, path, status):
        self.train.to_csv(f"{path}triplets/train_{status}_labels.csv", index=False, sep='\t')
        self.valid.to_csv(f"{path}/triplets/valid_{status}_labels.csv", index=False, sep='\t')
        self.test.to_csv(f"{path}/triplets/test_{status}_labels.csv", index=False, sep='\t')

@dataclasses.dataclass
class Nodes:
    train: Set = dataclasses.field(default_factory=set)
    valid: Set = dataclasses.field(default_factory=set)
    test: Set = dataclasses.field(default_factory=set)
    intersection : Set = dataclasses.field(default_factory=set)

def split(df_triplets, status, export, path):
    """
        Divide triplets into train, valid, test sets (80%, 10%, 10%)
        Return triplets and nodes in each set
        Status for random state
    """
    train, valid, test = np.split(df_triplets.sample(frac=1, random_state=status), [int(.8*len(df_triplets)), int(.9*len(df_triplets))])
    triplets = Triplets(train, valid, test)

    if export:
        triplets.export(path, status)

    nodes = Nodes()
    nodes.train = set(train["n1"]).union(set(train["n2"]))
    nodes.valid = set(valid["n1"]).union(set(valid["n2"]))
    nodes.test = set(test["n1"]).union(set(test["n2"]))
    nodes.intersection =  nodes.train.intersection(nodes.valid).intersection(nodes.test)

    return triplets, nodes

def filter_set(df, train):
    """
        Remove from test and valid set when head, rel or tail is not in train set
    """

    train_ent = set(train['n1'].tolist() + train['n2'].tolist())
    train_rel = set(train['t'].tolist())

    df = df[df['n1'].isin(train_ent)]
    df = df[df['n2'].isin(train_ent)]
    df = df[df['t'].isin(train_rel)]

    return df


if __name__ == '__main__':


    df_triplets = pd.read_csv("data/RLF/all_triplets_with_labels.txt", sep="\t", index_col=False)

    rlf, rlf_nodes = split(df_triplets, 20, True, RLF_PATH)

    filter_valid = filter_set(rlf.valid, rlf.train)
    filter_test = filter_set(rlf.test, rlf.train)

    rlf.train.to_csv("data/RLF/train_20_labels.txt", index=False, header=False, sep="\t")
    filter_valid.to_csv("data/RLF/valid_20_labels.txt", index=False, header=False, sep="\t")
    filter_test.to_csv("data/RLF/test_20_labels.txt", index=False, header=False, sep="\t")
