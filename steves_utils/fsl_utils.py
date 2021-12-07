from math import floor
from typing import Tuple
import torch
from torch import nn

# BEGIN KEEP IN THIS ORDER
# It's critical that you override this function first before importing the rest of easyfsl
def my_compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """ 
    Compute the dimension of the feature space defined by a feature extractor.
    Args:
        backbone: feature extractor

    Returns:
        shape of the feature vector computed by the feature extractor for an instance

    """
    input_images = torch.ones((4, 2, 128))
    output = backbone(input_images)

    return tuple(output.shape[1:])
import easyfsl.utils; easyfsl.utils.compute_backbone_output_shape = my_compute_backbone_output_shape
from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.methods import PrototypicalNetworks, AbstractMetaLearner
from easyfsl.utils import sliding_average, compute_backbone_output_shape
# END KEEP IN THIS ORDER


def split_ds_into_episodes(
    ds,
    n_way,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
    seed,
):
    """
    Splits ds into train,val,test sets, each being a episodic
    Returns:
        tuple(
            train_dl,
            val_list,
            test_list
        )
        Val and test are returned as lists so that they are not resampled dynamically
    """
    import copy 

    train_len = floor(len(ds)*0.7)
    val_len   = floor(len(ds)*0.15)
    test_len  = len(ds) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    train_ds.labels = [ex[1] for ex in train_ds]
    val_ds.labels   = [ex[1] for ex in val_ds]
    test_ds.labels  = [ex[1] for ex in test_ds]

    def wrap_in_dataloader(ds,n_tasks):
        sampler = TaskSampler(
                ds,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                n_tasks=n_tasks
            )

        return torch.utils.data.DataLoader(
            ds,
            num_workers=0,
            # persistent_workers=True,
            # prefetch_factor=50,
            # pin_memory=True,
            batch_sampler=sampler,
            collate_fn=sampler.episodic_collate_fn
        )

    # Deep copy necessary because of how tensors are retrieved from workers
    val_list = []
    for k in wrap_in_dataloader(val_ds, n_val_tasks):
        val_list.append(copy.deepcopy(k))

    test_list = []
    for k in wrap_in_dataloader(test_ds, n_test_tasks):
        test_list.append(copy.deepcopy(k))

    
    return (
        wrap_in_dataloader(train_ds, n_train_tasks),
        val_list,
        test_list,
    )