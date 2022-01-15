#! /usr/bin/env python3

from typing import List
import itertools
from torch.utils.data import DataLoader
from unittest import TestCase
import torch
from steves_utils.utils_v2 import (to_hash, norm)
from math import floor


def hash_episodic_dl(dl:DataLoader):
    hashes = []
    for u, (support_x, support_y, query_x, query_y, real_classes) in dl:
        hashes.append( to_hash(support_x) )
        hashes.append( to_hash(query_x) )
    
    return hashes

# Use to see if changing seed changes order, reproducability
def get_dls_identical(dl_a:DataLoader, dl_b:DataLoader):
    for a,b in zip(every_x_in_dl_generator(dl_a), every_x_in_dl_generator(dl_b)):
        if to_hash(a) != to_hash(b):
            return False
    
    return True




def every_x_in_dl_generator(dl:DataLoader):
    for u, (support_x, support_y, query_x, query_y, real_classes) in dl:
        for x in support_x:
            yield x
        for x in query_x:
            yield x




def test_correct_domains(self:TestCase, dl:DataLoader, expected_domains:List[int]):
    seen_domains = set()
    for u, ex in dl:
        seen_domains.add(u)
    self.assertEqual(
        seen_domains, set(expected_domains)
    )
                
def test_correct_labels(self:TestCase, dl:DataLoader, expected_labels:List[int]):
    seen_labels = set()
    for u, (support_x, support_y, query_x, query_y, real_classes) in dl:
        for y in torch.cat((support_y, query_y)):
            seen_labels.add(int(y))
    self.assertEqual(
        seen_labels, set(expected_labels)
    )


def test_correct_example_count_per_domain_per_label(self:TestCase, dl:DataLoader, expected_num:int):        
    examples_by_domain_by_label = {}

    for u, (support_x, support_y, query_x, query_y, real_classes) in dl:
        if u not in examples_by_domain_by_label:
            examples_by_domain_by_label[u] = {}
        for y in torch.cat((support_y, query_y)).numpy():
            if y not in examples_by_domain_by_label[u]: 
                examples_by_domain_by_label[u][y] = 0
            examples_by_domain_by_label[u][y] += 1

    print(examples_by_domain_by_label)
            
    for u, y_and_count in examples_by_domain_by_label.items():
        for y, count in y_and_count.items():
            self.assertEqual(count, expected_num)


def test_dls_disjoint(self:TestCase, dls:List[DataLoader]):
    dl_hashes = [set(hash_episodic_dl(dl)) for dl in dls]

    for a,b in itertools.product(dl_hashes):
        self.assertTrue(
            a.isdisjoint(b)
        )


def test_dls_equal(self:TestCase, dl_a:DataLoader, dl_b:DataLoader):
    self.assertTrue( get_dls_identical(dl_a, dl_b) )

def test_dls_notEqual(self:TestCase, dl_a:DataLoader, dl_b:DataLoader):
    self.assertFalse( get_dls_identical(dl_a, dl_b) )


def test_len(self:TestCase, dl:DataLoader):
    expected_len = len(dl)
    calc_len = sum([1 for _ in dl])

    self.assertEqual(
        calc_len,
        expected_len
    )

def test_shape(self:TestCase, dl:DataLoader, n_way, n_shot, n_query):
    for u, (support_x, support_y, query_x, query_y, real_classes) in dl:
        self.assertEqual( support_x.numpy().shape   ,(n_way*n_shot, 2, 256) )
        self.assertEqual( support_y.numpy().shape   ,(n_way*n_shot,) )
        self.assertEqual( query_x.numpy().shape ,(n_way*n_query, 2, 256) )
        self.assertEqual( query_y.numpy().shape ,(n_way*n_query,) )



def test_splits(self:TestCase, dls:List[DataLoader], ratios:List[float]):
    lens = [len(dl) for dl in dls ]

    calculated_ratios = [l/sum(lens) for l in lens]

    for expected, calculated in zip(ratios, calculated_ratios):
        self.assertAlmostEqual(expected, calculated)


def test_episodes_have_no_repeats(self:TestCase, dl:DataLoader):
    hashes = hash_episodic_dl(dl)

    self.assertEqual(
        len(hashes),
        len(set(hashes))
    )

def test_approximate_number_episodes(
    self:TestCase,
    dl:DataLoader,
    k_factor,
    num_examples_per_domain_per_label,
    num_classes,
    num_domains,
    n_way,n_shot,n_query
    ):
    expected_episodes = floor(k_factor * num_examples_per_domain_per_label * num_classes * num_domains / (n_way*(n_shot + n_query)))

    self.assertAlmostEqual(
        expected_episodes,
        len(dl)
    )

def test_normalization(self:TestCase, non_normalized_dl:DataLoader, norm_algos:List[str]):       
    for algo in norm_algos:
        for x in every_x_in_dl_generator(non_normalized_dl):
            self.assertNotEqual(
                x,
                norm(x, algo)
            )