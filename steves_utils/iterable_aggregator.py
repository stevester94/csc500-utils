#! /usr/bin/env python3

import numpy as np

class Iterable_Aggregator:
    """
    Combine iterables into one big iterable. Optionally randomize by passing randomizer_seed
    In the case of None randomizer_seed, the iterables will be next()'d in order (and therefore order matters on the input)
    """
    def __init__(self, iterables, randomizer_seed=None) -> None:
        self.iterables = iterables

        for i in self.iterables:
            iter_method = getattr(i, "__iter__", None)
            if not callable(iter_method):
                raise Exception("Received a non-iterable object")
        
        self.lengths = [len(l) for l in self.iterables]
        self.length = sum(self.lengths)

        if randomizer_seed is not None:
            self.rng = np.random.default_rng(randomizer_seed)
        else:
            self.rng = None
        
        self.iterators = []


    def __iter__(self):
        self.iterators = [iter(i) for i in self.iterables]

        return self

    def __next__(self):
        """
        This one is a little funky. We randomly choose an iterator and return its next()
        If that iterator is empty, we remove it from our list of iterators, then 
        we call next() on ourself, so the process repeats recursively.
        The base case of the recursion is either we find an iterator that has not been
        exhausted, or we remove all iterators and raise StopIteration
        """
        if self.iterators == []:
            raise StopIteration

        # Select a random iterator if we have rng, otherwise just pick the next one
        if self.rng is not None:
            it = self.rng.integers(0, len(self.iterators), 1)[0]
            it = self.iterators[it]
        else:
            it = self.iterators[0]

        try:
            x = next(it)
        except StopIteration:
            self.iterators.remove(it)
            x = next(self)
            
        return x

    def __len__(self):
        return self.length
    



if __name__ == "__main__":
    import unittest

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        def test_non_randomized_aggregation(self):
            lists = [
                list(range(10)),
                list(range(10, 20)),
                [],
                range(12),
                "abc"
            ]

            ground_truth = []
            for l in lists:
                ground_truth.extend(l)

            # ia = Iterable_Aggregator(lists, 1337)
            ia = Iterable_Aggregator(lists)

            self.assertEqual(
                list(ia),
                ground_truth
            )

        def test_randomized_aggregation(self):
            lists = [
                list(range(10)),
                list(range(10, 20)),
                [],
                range(12),
                "abc"
            ]

            ground_truth = []
            for l in lists:
                ground_truth.extend(l)

            ia = Iterable_Aggregator(lists, 1337)

            self.assertNotEqual(
                list(ia),
                ground_truth
            )

            # Visual inspection
            print(list(ia))
            print(ground_truth)
            input("Are the above not equal?")

        def test_determinism(self):
            lists = [
                list(range(10)),
                list(range(10, 20)),
                [],
                range(12),
                "abc"
            ]

            ground_truth = []
            for l in lists:
                ground_truth.extend(l)

            ia_1 = Iterable_Aggregator(lists, 1337)
            ia_2 = Iterable_Aggregator(lists, 1337)
            ia_3 = Iterable_Aggregator(lists, 420)

            self.assertEqual(
                list(ia_1),
                list(ia_2)
            )

            self.assertNotEqual(
                list(ia_1),
                list(ia_3)
            )
        
        def test_len(self):
            lists = [
                list(range(10)),
                list(range(10, 20)),
                [],
                range(12),
                "abc"
            ]

            ground_truth = []
            for l in lists:
                ground_truth.extend(l)

            ia_1 = Iterable_Aggregator(lists, 1337)

            self.assertEqual(
                len(ia_1),
                len(ground_truth)
            )


    # class test_File_As_Windowed_Sequence(unittest.TestCase):
    #     @classmethod
    #     def setUpClass(self) -> None:
    #         self.sequences = []

    #         for i in range(NUM_SUBSEQUENCES):
    #             length = np.random.default_rng().integers(MIN_SUBSEQ_LENGTH+1, MAX_SUBSEQ_LENGTH, 1)[0]

    #             if len(self.sequences) > 0:
    #                 last_seq_end = self.sequences[-1][-1]
    #             else:
    #                 last_seq_end = 0

    #             self.sequences.append(list(range(last_seq_end+1, last_seq_end+1+length)))
    #         # print(self.sequences)
    #     # @classmethod
    #     # def tearDownClass(self) -> None:
    #     #     pass

    #     # Verify our test bed is good
    #     def test_testbed_is_kosher(self):
    #         self.assertEqual(len(self.sequences), NUM_SUBSEQUENCES)

    #         for i in self.sequences:
    #             self.assertTrue(len(i) < MAX_SUBSEQ_LENGTH)
    #             self.assertTrue(len(i) > MIN_SUBSEQ_LENGTH)

    #         consolidated = []
    #         for i in self.sequences:
    #             consolidated.extend(i)

    #         self.assertEqual(list(range(1,len(consolidated)+1)), consolidated)

    #     def test_length(self):
    #         sa =  Sequence_Aggregator(self.sequences)

    #         consolidated = []
    #         for i in self.sequences:
    #             consolidated.extend(i)

    #         self.assertEqual(len(consolidated), len(sa))


    #     def test_simple_aggregation(self):
    #         sa =  Sequence_Aggregator(self.sequences)

    #         target = range(1, len(sa) + 1)

    #         for i in range(len(sa)):
    #             self.assertEqual(sa[i], target[i])

    #     def test_full_length_indexing(self):
    #         sa =  Sequence_Aggregator(self.sequences)

    #         for i in range(len(sa)):
    #             _ = sa[i]

    #     def test_iteration(self):
    #         sa =  Sequence_Aggregator(self.sequences)

    #         target = range(1, len(sa) + 1)

    #         for i,x in enumerate(sa):
    #             self.assertEqual(x, target[i])


    #     def test_iter(self):
    #         sa =  Sequence_Aggregator(self.sequences)
    #         it = iter(sa)

    #         for i in range(len(sa)):
    #             _ = next(it)
            

    unittest.main()
