import pickle

class Stratified_Dataset:
    """Convenience wrapper around the stratified dataset pickled dictionary"""
    def __init__(
        self,
        pickle_path:str
    ) -> None:
        with open(pickle_path, "rb") as f:
            self.d = dict(pickle.load(f))
    
        if "data" not in self.d.keys():
            raise Exception("data key is not in pickle")

        if "metadata" not in self.d.keys():
            raise Exception("metadata key is not in pickle")

    def get_metadata(self)->dict:
        return self.d["metadata"]

    def get_data(self)->dict:
        return self.d["data"]