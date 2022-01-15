from steves_utils.CORES.utils import node_name_to_id

def genericize_stratified_dataset(
    sds:dict,
    domains:list,
    labels:list,
    n_per_u_per_y:int,
    )->dict:
    gsd = {}

    for domain in domains:
        gsd[domain] = {}
        for label in labels:
            # The X array is randomized already, so we can do a simple truncation
            gsd[domain][node_name_to_id(label)] = sds[domain][label][:n_per_u_per_y]

    return gsd