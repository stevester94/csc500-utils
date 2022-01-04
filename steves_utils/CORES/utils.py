#! /usr/bin/env python3
import sys
import numpy as np
import pickle
import os

from torch.utils import data
# from definitions import *
import steves_utils.utils_v2 as steves_utils_v2
import steves_utils.torch_utils as steves_torch_utils
from typing import List

ALL_DAYS = [
        1,
        2,
        3,
        4,
        5,
]

dataset_day_name_mapping = {
    1: 'grid_2019_12_25.pkl',
    2: 'grid_2020_02_03.pkl',
    3: 'grid_2020_02_04.pkl',
    4: 'grid_2020_02_05.pkl',
    5: 'grid_2020_02_06.pkl',
}

# We limit ourselves to the nodes that exist in all days of the dataset
ALL_NODES = [
    '1-10.', '1-11.', '1-15.', '1-16.', '1-17.', '1-18.', '1-19.', '10-4.', '10-7.', '11-1.', '11-14.', '11-17.', 
    '11-20.', '11-7.', '13-20.', '13-8.', '14-10.', '14-11.', '14-14.', '14-7.', '15-1.', '15-20.', '16-1.', 
    '16-16.', '17-10.', '17-11.', '17-2.', '19-1.', '19-16.', '19-19.', '19-20.', '19-3.', '2-10.', '2-11.', 
    '2-17.', '2-18.', '2-20.', '2-3.', '2-4.', '2-5.', '2-6.', '2-7.', '2-8.', '3-13.', '3-18.', '3-3.', '4-1.', 
    '4-10.', '4-11.', '4-19.', '5-5.', '6-15.', '7-10.', '7-14.', '8-18.', '8-20.', '8-3.', '8-8.'
]

# Obtained from get_nodes_with_a_minimum_num_examples_for_each_day(1000)
ALL_NODES_MINIMUM_1000_EXAMPLES = [
    '17-11.', '10-7.', '8-20.', '14-7.', '19-1.', '7-14.', '3-13.', 
    '15-1.', '4-1.', '19-19.', '5-5.', '15-20.', '13-8.', '11-1.', '2-6.', '8-3.', '16-16.', '6-15.'
]


ALL_NODES_INDICES = {
    node_name: ALL_NODES.index(node_name) for node_name in ALL_NODES
}

def get_cores_dataset_path():
    return os.path.join(steves_utils_v2.get_datasets_base_path(), "CORES/orbit_rf_identification_dataset_updated")

def load_from_disk(day:int, root_dir:str=get_cores_dataset_path()) -> dict:
    if day not in dataset_day_name_mapping:
        raise ValueError(f"day {day} not valid")

    dataset_path = os.path.join(root_dir, dataset_day_name_mapping[day])
    with open(dataset_path,'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_single_day(day:int, nodes_to_get, root_dir:str=get_cores_dataset_path()) -> list:
    """
    Args:
        day: the day of collects to get
        nodes_to_get: Which nodes to get, by name
        root_dir: where the experiment lives
    Returns:
        List of dicts, one dict for each example
    """
    dataset = load_from_disk(day, root_dir)
    
    d = []

    for node_name, node_data in zip(dataset["node_list"], dataset["data"]):
        if node_name in nodes_to_get:
            for capture in node_data:
                example = {
                    "node_name": node_name,
                    "day": day,
                    "IQ": capture
                }

                d.append(example)

    return d

def get_dataset_by_day_and_node_name(
    days_to_get:List[int],
    num_examples_per_node_per_day,
    nodes_to_get,
    seed:int,
    root_dir:str=get_cores_dataset_path()) -> list:
    """
    Get list of examples matching criteria
    """
    

    days = { day:{} for day in days_to_get }

    # Put each example in a nested dict days[day][node_name]
    # Only get the days and nodes we actually want
    for day in days_to_get:
        dataset = get_single_day(day=day, nodes_to_get=nodes_to_get, root_dir=root_dir)
        for ex in dataset:
            if ex["node_name"] in nodes_to_get:
                if ex["node_name"] not in days[day]:
                    days[day][ex["node_name"]] = []
                days[day][ex["node_name"]].append(ex)

    # Validate each device has our minimum amount of examples       
    for day in days_to_get:
        for node_name, examples in days[day].items():
            if len(examples) < num_examples_per_node_per_day:
                raise RuntimeError(f"{node_name} on day {day} only has {len(examples)} examples, but we requested {num_examples_per_node_per_day}")


    # randomly sample the requested number of examples for each [day][node_name][examples]
    for day, node in days.items():
        for node_name, examples in node.items():
            # Never got around to testing this, but the intent here is that each device's sampling is random AND independent of
            # whatever each other device/day we've requested
            rng = np.random.default_rng(seed + len(examples))
            days[day][node_name] = rng.choice(examples, size=num_examples_per_node_per_day, replace=False)

    # put all [day][node_name][examples] into one big list
    dataset = []
    for day, node in days.items():
        for node_name, examples in node.items():
            dataset.extend(examples)

    # Shuffle the list
    rng = np.random.default_rng(seed)
    rng.shuffle(dataset)

    return dataset

def split_dataset_by_day(dataset:list):
    days = {}

    for ex in dataset:
        day = None
        if isinstance(ex, dict):
            day = ex["day"]
        if isinstance(ex, tuple):
            day = ex[2]

        if day not in days:
            days[day] = []
        days[day].append(ex)
    
    return days

def split_dataset_by_node_name(dataset:list):
    nodes = {}

    for ex in dataset:
        node = None
        if isinstance(ex, dict):
            node = ex["node_name"]
        if isinstance(ex, tuple):
            node = ex[1]

        if node not in nodes:
            nodes[node] = []
        nodes[node].append(ex)
    
    return nodes

def tupleify_dataset(dataset:list):
    return list(
        map(
            lambda ex: (ex["IQ"], ALL_NODES_INDICES[ex["node_name"]], ex["day"]),
            dataset
        )
    )

def reshape_tupleified_dataset_iq(dataset:list)->list:
    """
    It's just a transpose since the data is originally saved as "rows"
    IE shape is originally 256,2
    But we want 2,256
    """
    return list(
        map(
            lambda ex: (ex[0].T, ex[1], ex[2]),
            dataset
        )
    )


def split_dataset_and_group_by_day(dataset:list, seed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train, val, test = steves_torch_utils.split_dataset_by_percentage_v2(
        train=train_ratio,
        val=val_ratio,
        dataset=dataset,
        seed=seed
    )

    train = split_dataset_by_day(train)
    val = split_dataset_by_day(val)
    test = split_dataset_by_day(test)

    return (train, val, test)

def get_it(
    days_to_get:List[int],
    num_examples_per_node_per_day,
    nodes_to_get,
    seed:int,
    root_dir:str=get_cores_dataset_path()
    ) -> tuple:
    """
    This is the one to use
    """

    dataset = get_dataset_by_day_and_node_name(
        days_to_get=days_to_get,
        num_examples_per_node_per_day=num_examples_per_node_per_day,
        nodes_to_get=nodes_to_get,
        seed=1337,
        root_dir=root_dir
    )
    dataset = tupleify_dataset(dataset)
    dataset = reshape_tupleified_dataset_iq(dataset)
    dataset = split_dataset_and_group_by_day(dataset, seed)

    return dataset


def get_nodes_with_a_minimum_num_examples_for_each_day(num_examples_per_node_per_day):
    days = {
        day: {} for day in ALL_DAYS
    }

    # Get counts for each device for each day
    for day in ALL_DAYS:
        dataset = get_single_day(day, nodes_to_get=ALL_NODES)
        for ex in dataset:
            if ex["node_name"] not in days[day]:
                days[day][ex["node_name"]] = 0
            days[day][ex["node_name"]] += 1

    # Remove any devices that are below a certain threshold on each day        
    for day in ALL_DAYS:
        not_accepted = []
        for node_name, count in days[day].items():
            if count < num_examples_per_node_per_day:
                not_accepted.append(node_name)

        for node_name in not_accepted: del days[day][node_name]
    
    for day in days.keys():
        node_names = days[day].keys()
        days[day] = set(node_names)

    days = days[1].intersection(
        days[2],
        days[3],
        days[4],
        days[5],
    )

    return list(days)



def test_examples_equal(a,b):
    return (
        np.array_equal(a[0], b[0]) and
        a[1] == b[1] and
        a[2] == b[2]
    )

import unittest
class Test_get_it(unittest.TestCase):
    def test_correct_days(self):
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=10,
            nodes_to_get=ALL_NODES,
            seed=1337,
        )

        for ds in datasets:
            for day, examples in ds.items():
                for ex in examples:
                    self.assertEqual(
                        day, ex[2]
                    )

    def test_correct_num_day_examples(self):
        NUM_EXAMPLES_PER_NODE_PER_DAY = 10
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)

        fuck = {
            day: {
                dev: 0 for dev in range(len(ALL_NODES))
            } for day in ALL_DAYS
        }

        for ex in all:
            fuck[ex[2]][ex[1]] += 1
            
        for day, d in fuck.items():
            for dev, count in d.items():
                self.assertEqual( count, NUM_EXAMPLES_PER_NODE_PER_DAY)


    def test_correct_num_day_examples_again(self):
        NUM_EXAMPLES_PER_NODE_PER_DAY = 1000
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)

        fuck = {
            day: {
                dev: 0 for dev in [ALL_NODES_INDICES[node] for node in ALL_NODES_MINIMUM_1000_EXAMPLES]
            } for day in ALL_DAYS
        }

        for ex in all:
            fuck[ex[2]][ex[1]] += 1
            
        for day, d in fuck.items():
            for dev, count in d.items():
                self.assertEqual( count, NUM_EXAMPLES_PER_NODE_PER_DAY)

    @unittest.expectedFailure
    def test_correct_num_day_examples_again(self):
        NUM_EXAMPLES_PER_NODE_PER_DAY = 1500
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)

        fuck = {
            day: {
                dev: 0 for dev in [ALL_NODES_INDICES[node] for node in ALL_NODES_MINIMUM_1000_EXAMPLES]
            } for day in ALL_DAYS
        }

        for ex in all:
            fuck[ex[2]][ex[1]] += 1
            
        for day, d in fuck.items():
            for dev, count in d.items():
                self.assertEqual( count, NUM_EXAMPLES_PER_NODE_PER_DAY)

    def test_seed_same(self):
        NUM_EXAMPLES_PER_NODE_PER_DAY = 1000
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)
        
        all_1 = all

        NUM_EXAMPLES_PER_NODE_PER_DAY = 1000
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)

        all_2 = all

        for a,b in zip(all_1, all_2):
            self.assertTrue(
                test_examples_equal(a,b)
            )

    @unittest.expectedFailure
    def test_seed_different(self):
        NUM_EXAMPLES_PER_NODE_PER_DAY = 1000
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1337,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)
        
        all_1 = all

        NUM_EXAMPLES_PER_NODE_PER_DAY = 1000
        datasets = get_it(
            days_to_get=ALL_DAYS,
            num_examples_per_node_per_day=NUM_EXAMPLES_PER_NODE_PER_DAY,
            nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
            seed=1338,
        )

        all = []

        for ds in datasets:
            for day, examples in ds.items():
                all.extend(examples)

        all_2 = all

        for a,b in zip(all_1, all_2):
            self.assertTrue(
                test_examples_equal(a,b)
            )

if __name__ == "__main__":
    unittest.main()

    # Testing purposes
    # train, _, _ = get_it(
    #     days_to_get=ALL_DAYS,
    #     num_examples_per_node_per_day=100,
    #     nodes_to_get=ALL_NODES_MINIMUM_1000_EXAMPLES,
    #     seed=1337,
    # )

    # x,y,u = train[1][0]

    # print(x.shape)
