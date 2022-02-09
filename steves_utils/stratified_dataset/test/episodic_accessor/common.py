import unittest
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

from steves_utils.stratified_dataset.episodic_test_cases import(
    test_correct_domains,
    test_correct_labels,
    test_correct_example_count_per_domain_per_label,
    test_dls_disjoint,
    test_dls_equal,
    test_dls_notEqual,
    test_len,
    test_splits,
    test_episodes_have_no_repeats,
    test_shape,
    test_approximate_number_episodes,
    test_no_duplicates_in_dl,
    hash_episodic_dl
)

from steves_utils.stratified_dataset.episodic_accessor import Episodic_Accessor_Factory

class Test_Episodic_Accessor(unittest.TestCase):
    domains= None
    labels= None
    num_examples_per_domain_per_label= None
    dataset_seed= None
    iterator_seed= None
    n_shot= None
    n_way= None
    n_query= None
    train_val_test_k_factors= None
    train_val_test_percents= None
    pickle_path = None

    @classmethod
    def setUpClass(cls):

        eaf = Episodic_Accessor_Factory(
            labels=cls.labels,
            domains=cls.domains,
            num_examples_per_domain_per_label=cls.num_examples_per_domain_per_label,
            pickle_path=cls.pickle_path,
            dataset_seed=cls.dataset_seed,
            n_shot=cls.n_shot,
            n_way=cls.n_way,
            n_query=cls.n_query,
            iterator_seed=cls.iterator_seed,
            train_val_test_k_factors=cls.train_val_test_k_factors,
            train_val_test_percents=cls.train_val_test_percents,
        )

        cls.TRAIN, cls.VAL, cls.TEST = eaf.get_train(), eaf.get_val(), eaf.get_test()

        cls.ALL_DL = (cls.TRAIN, cls.VAL, cls.TEST)

        cls.generic_labels = [cls.labels.index(y) for y in cls.labels]

    def test_correct_domains(self):
        for dl in self.ALL_DL:
            test_correct_domains(self, dl, self.domains)

    def test_correct_labels(self):
        for dl in self.ALL_DL:
            test_correct_labels(self, dl, self.generic_labels)


    def test_correct_example_count_per_domain_per_label(self):
        for dl,ratio in zip(self.ALL_DL[:1], self.train_val_test_percents[:1]):
            test_correct_example_count_per_domain_per_label(self, dl, int(self.num_examples_per_domain_per_label*ratio))


    def test_dls_disjoint(self):
        test_dls_disjoint(self, self.ALL_DL)

    
    def test_shape(self):
        for dl in self.ALL_DL:
            test_shape(self, dl, self.n_way, self.n_shot, self.n_query)

    def test_repeatability(self):
        np.random.seed(self.iterator_seed)
        random.seed(self.iterator_seed)
        torch.manual_seed(self.iterator_seed)
        eaf = Episodic_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            pickle_path=self.pickle_path,
            dataset_seed=self.dataset_seed,
            n_shot=self.n_shot,
            n_way=self.n_way,
            n_query=self.n_query,
            iterator_seed=self.iterator_seed,
            train_val_test_k_factors=self.train_val_test_k_factors,
            train_val_test_percents=self.train_val_test_percents,
        )

        TRAIN, VAL, TEST = eaf.get_train(), eaf.get_val(), eaf.get_test()
        
        NUM_ITERATIONS = 5
        first_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                first_h.append(hash(tuple(hash_episodic_dl(dl))))


        np.random.seed(self.iterator_seed)
        random.seed(self.iterator_seed)
        torch.manual_seed(self.iterator_seed)
        eaf = Episodic_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            pickle_path=self.pickle_path,
            dataset_seed=self.dataset_seed,
            n_shot=self.n_shot,
            n_way=self.n_way,
            n_query=self.n_query,
            iterator_seed=self.iterator_seed,
            train_val_test_k_factors=self.train_val_test_k_factors,
            train_val_test_percents=self.train_val_test_percents,
        )

        TRAIN, VAL, TEST = eaf.get_train(), eaf.get_val(), eaf.get_test()

        second_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                second_h.append(hash(tuple(hash_episodic_dl(dl))))

        self.assertEqual(
            first_h,
            second_h
        )


    def test_seed_changes(self):
        np.random.seed(self.iterator_seed)
        random.seed(self.iterator_seed)
        torch.manual_seed(self.iterator_seed)
        eaf = Episodic_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            pickle_path=self.pickle_path,
            dataset_seed=self.dataset_seed,
            n_shot=self.n_shot,
            n_way=self.n_way,
            n_query=self.n_query,
            iterator_seed=self.iterator_seed,
            train_val_test_k_factors=self.train_val_test_k_factors,
            train_val_test_percents=self.train_val_test_percents,
        )

        TRAIN, VAL, TEST = eaf.get_train(), eaf.get_val(), eaf.get_test()
        
        NUM_ITERATIONS = 5
        first_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                first_h.append(hash(tuple(hash_episodic_dl(dl))))


        np.random.seed(self.iterator_seed)
        random.seed(self.iterator_seed)
        torch.manual_seed(self.iterator_seed)
        eaf = Episodic_Accessor_Factory(
            labels=self.labels,
            domains=self.domains,
            num_examples_per_domain_per_label=self.num_examples_per_domain_per_label,
            pickle_path=self.pickle_path,
            dataset_seed=self.dataset_seed,
            n_shot=self.n_shot,
            n_way=self.n_way,
            n_query=self.n_query,
            iterator_seed=self.iterator_seed+1,
            train_val_test_k_factors=self.train_val_test_k_factors,
            train_val_test_percents=self.train_val_test_percents,
        )

        TRAIN, VAL, TEST = eaf.get_train(), eaf.get_val(), eaf.get_test()

        second_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                second_h.append(hash(tuple(hash_episodic_dl(dl))))

        self.assertNotEqual(
            first_h,
            second_h
        )

    def test_train_changes_between_iterations(self):
        test_dls_notEqual(self, self.TRAIN, self.TRAIN)

    def test_val_and_test_dont_change_between_iterations(self):
        test_dls_equal(self, self.VAL, self.VAL)
        test_dls_equal(self, self.TEST, self.TEST)
        test_dls_notEqual(self, self.VAL, self.TEST)

    def test_approximate_number_episodes(self):
        for i,dl in enumerate(self.ALL_DL):
            if i == 0: continue
            test_approximate_number_episodes(
                self,
                dl,
                self.train_val_test_k_factors[i],
                int(self.num_examples_per_domain_per_label*self.train_val_test_percents[i]),
                len(self.labels),
                len(self.domains),
                self.n_way,
                self.n_shot,
                self.n_query,
            )
    
    def test_len(self):
        for dl in self.ALL_DL:
            test_len(self, dl)

    def test_episodes_have_no_repeats(self):
        for i,dl in enumerate(self.ALL_DL):
            test_episodes_have_no_repeats(self, dl)
    
    def test_splits(self):
        test_splits(self, self.ALL_DL, self.train_val_test_percents, self.train_val_test_k_factors)

    def test_no_duplicates_in_dl(self):
        for i,dl in enumerate(self.ALL_DL):
            test_no_duplicates_in_dl(self, dl)
