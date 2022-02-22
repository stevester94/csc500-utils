#! /usr/bin/env python3

from functools import reduce
import numpy as np
import torch

"""
Returns a lambda which in turn calls all lambda functions in <lambdas> in ascending order

Let lambdas = [F(x), G(x)]
Then chain_lambdas returns H(x), where H(x) = F((G(x))
"""
def chain_lambdas(lambdas:list):
    return lambda origin: reduce((lambda x, y: y(x)), lambdas, origin)


def get_average_magnitude(x):
    assert x.shape[0] == 2
    x_t = x.T
    mag = np.linalg.norm(x_t, axis = 1)
    assert np.isclose(mag[0], np.sqrt(np.sum(x_t[0]**2)))
    
    return np.mean(mag)
    
def normalize_to_unit_magnitude(x):
    return x/get_average_magnitude(x)

def get_average_power(x):
    assert x.shape[0] == 2
    x_t = x.T
    power = np.linalg.norm(x_t, axis = 1)**2
    
    assert np.isclose(power[0], np.sum(x_t[0]**2))
    
    return np.mean(power)
    
def normalize_to_unit_power(x):
    return x/np.sqrt(get_average_power(x))


def normalize(sig_u, norm_type:str):
    if isinstance(sig_u, torch.Tensor):
        x = sig_u.numpy()
    else:
        x = sig_u

    if norm_type == "unit_mag":
        ret = normalize_to_unit_magnitude(x)
    elif norm_type == "unit_power":
        ret = normalize_to_unit_power(x)
    else:
        raise Exception(f"Unknown norm_type: {norm_type}")


    if isinstance(sig_u, torch.Tensor):
        ret = torch.from_numpy(ret)
    
    return ret

if __name__ == "__main__":
    import unittest

    l = [
        [3,4, -2],
        [2,5, -3]
    ]
    l = np.array(l)

    assert l.shape[0] == 2

    class Test_Chain_Lambdas(unittest.TestCase):
        def test_ordering(self):
            l1 = lambda x: x*2
            l2 = lambda x: x-2

            c1 = chain_lambdas([l1,l2])
            c2 = chain_lambdas([l2, l1])

            self.assertEqual(c1(2),2)
            self.assertEqual(c2(2),0)

    class Test_Magnitude(unittest.TestCase):
        def test_get_average_magnitude(self):
            self.assertAlmostEqual(
                get_average_magnitude(l),
                np.mean(
                    np.array([
                        np.sqrt(l[0][0]**2 + l[1][0]**2),
                        np.sqrt(l[0][1]**2 + l[1][1]**2),
                        np.sqrt(l[0][2]**2 + l[1][2]**2),
                    ])
                )
            )

        def normalize_to_unit_magnitude(self):
            l_norm = normalize_to_unit_magnitude(l)
            self.assertEqual(
                get_average_magnitude(l_norm),
                1
            )

    class Test_Power(unittest.TestCase):
        def test_get_average_power(self):
            self.assertAlmostEqual(
                get_average_power(l),
                np.mean(
                    np.array([
                        l[0][0]**2 + l[1][0]**2,
                        l[0][1]**2 + l[1][1]**2,
                        l[0][2]**2 + l[1][2]**2,
                    ])
                )
            )

        def normalize_to_unit_power(self):
            l_norm = normalize_to_unit_magnitude(l)
            self.assertEqual(
                get_average_magnitude(l_norm),
                1
            )



    

    unittest.main()