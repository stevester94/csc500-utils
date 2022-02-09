from random import choice
from steves_utils.stratified_dataset.stratified_dataset import Stratified_Dataset
import numpy as np

def filter_sds_in_place(
    sds:Stratified_Dataset,
    domains:list,
    labels:list,
    num_examples_per_domain_per_label:int,
    seed,
    )->None:
    """Delete any domains and labels we dont want, shuffle the X, and take only num_examples_per_domain_per_label

    All done in place
    """
    data = sds.get_data()

    rng = np.random.default_rng(seed)
    

    labels_as_ints = {y: labels.index(y) for y in labels}

    for u in list(data.keys()):
        if u not in domains: del data[u]
    
    assert set(list(data.keys())) == set(domains)

    for u, y_X_dict in data.items():
        for y in list(y_X_dict.keys()):
            if y not in labels:
                del y_X_dict[y]
    
    for u, y_X_dict in data.items():
        assert set(list(y_X_dict.keys())) == set(labels)

    """
    It's a little ugly looking, but we're renaming keys, and then taking a subsample of the examples
    """
    for u, y_X_dict in data.items():
        for y in list(y_X_dict.keys()):
            new_label = labels_as_ints[y]

            y_X_dict[new_label] = y_X_dict.pop(y)
            data[u][new_label] = rng.choice(data[u][new_label], num_examples_per_domain_per_label, False)


    assert set(list(data.keys())) == set(domains)
    for u, y_X_dict in data.items():
            assert set(list(y_X_dict.keys())) == set([labels_as_ints[y] for y in labels])
            for y, X in y_X_dict.items():
                assert len(X) == num_examples_per_domain_per_label