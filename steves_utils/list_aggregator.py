#! /usr/bin/env python3

import numpy as np

class Sequence_Aggregator:
    """
    Combine input sequences into one large sequence.
    Indices are only preserved for the first sequence, the rest are the sum of the previous sequence.
    Order matters for sequence.
    Each list in sequence must have length > zero
    """
    def __init__(self, sequences) -> None:
        self.sequences = sequences

        for l in self.sequences:
            if len(l) <= 0:
                raise "List received in List_Aggregator of 0 or negative length"
        
        self.lengths = [len(l) for l in self.sequences]
    
    def __getitem__(self, idx):
        return None

    def __iter__(self):
        return self

    def __next__(self):
        for s in self.sequences:
            for i in s:
                return s