#! /usr/bin/env python3
import pickle
import os

from steves_utils.utils_v2 import (
    get_datasets_base_path
)


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
    '10-7.', '11-1.', '13-8.', '14-7.', '15-1.', '15-20.', '16-16.', '17-11.', '19-1.', 
    '19-19.', '2-6.', '3-13.', '4-1.', '5-5.', '6-15.', '7-14.', '8-20.', '8-3.'
]



ALL_NODES_INDICES = {
    node_name: ALL_NODES.index(node_name) for node_name in ALL_NODES
}

def node_name_to_id(node_name:str)->int:
    return ALL_NODES_INDICES[node_name]

def get_cores_dataset_path():
    return os.path.join(get_datasets_base_path(), "CORES/orbit_rf_identification_dataset_updated")

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

    days = list(days)
    days.sort()

    return days


if __name__ == "__main__":
    nodes = get_nodes_with_a_minimum_num_examples_for_each_day(1000)
    nodes.sort()
    ALL_NODES_MINIMUM_1000_EXAMPLES.sort()

    print(nodes)
    print(ALL_NODES_MINIMUM_1000_EXAMPLES)
