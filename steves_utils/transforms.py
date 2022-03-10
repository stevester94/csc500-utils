#! /usr/bin/env python3

from functools import reduce
import numpy as np
import torch

from scipy import signal
from scipy.signal import butter, lfilter, freqz


"""
Returns a lambda which in turn calls all lambda functions in <lambdas> in descending order

Let lambdas = [F(x), G(x)]
Then chain_lambdas returns H(x), where H(x) = F((G(x))
"""
def chain_lambdas(lambdas:list):
    return lambda origin: reduce((lambda x, y: y(x)), reversed(lambdas), origin)


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

def resample(x, orig_f_Hz, target_f_Hz):
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x = x.numpy()

    # x = np.apply_along_axis(lambda args: [complex(*args)], 1, x.T).flatten()
    x = x[0] + x[1]*1j
    x = signal.resample(x, int(len(x)*target_f_Hz/orig_f_Hz))
    x = np.array([np.real(x), np.imag(x)])

    if is_tensor:
        x = torch.from_numpy(x)

    return x


def jitter(x, total_length, stddev):
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x = x.numpy()

    assert x.shape[1] < total_length

    u = (total_length - x.shape[1]) / 2
    u_std_dev = stddev

    CRISIS_AVERSION_COUNTER = 100
    while True:
        a = np.random.default_rng().normal(u, u_std_dev)
        a = int(np.round(a))

        b = total_length - x.shape[1] - a

        if a > 1 and b > 1:
            break
        CRISIS_AVERSION_COUNTER -= 1

        if CRISIS_AVERSION_COUNTER == 0:
            raise Exception("Infinite loop detected in jitter function")

    a = np.zeros([2,a], dtype=x.dtype)
    b = np.zeros([2,b], dtype=x.dtype)

    x = np.concatenate(
        (a,x,b), axis=1
    )

    if x.shape[1] != total_length:
        raise Exception(f"jitter failed, created an array with shape {x.shape}")
    assert x.shape[1] == total_length

    if is_tensor:
        x = torch.from_numpy(x)
    
    return x

    


filters = {} # order, fs, cutoff

def filter_signal(x, order, fs, cutoff):
    def build_butter_filter(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_filter(b,a,x):
        y = lfilter(b, a, x)
        return y

    if (order, fs, cutoff) not in filters:
        filters[(order, fs, cutoff)] = build_butter_filter(cutoff=cutoff, fs=fs, order=order)
    
    f = filters[(order, fs, cutoff)]

    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x = x.numpy()
    
    x = apply_filter(f[0], f[1], x)

    if is_tensor:
        x = torch.from_numpy(x)
    
    return x



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


"""
Returns a lambda which in turn calls all functions corresponding to the string in <transforms> in descending order

Let lambdas = [F(x), G(x)]
Then chain_lambdas returns H(x), where H(x) = F((G(x))
"""
def get_chained_transform(transforms:list):
    lambdas = []
    for t in transforms:
        if t == "unit_mag": lambdas.append( lambda x: normalize(x, "unit_mag"))
        elif t == "unit_power": lambdas.append( lambda x: normalize(x, "unit_power"))
        elif t == "times_two": lambdas.append( lambda x: x*2)
        elif t == "minus_two": lambdas.append( lambda x: x-2)
        elif t == "times_zero": lambdas.append( lambda x: x*0)
        elif t == "take_160": lambdas.append( lambda x: x[:, :160])
        elif t == "take_200": lambdas.append( lambda x: x[:, :200])
        elif t == "resample_20Msps_to_25Msps": lambdas.append( lambda x: resample(x, 20e6, 25e6))
        elif t == "lowpass_+/-10MHz": lambdas.append( lambda x: filter_signal(x, 15, 25e6, 10e6))
        elif t == "jitter_256_10": lambdas.append( lambda x: jitter(x, 256, 10))
        elif t == "jitter_256_5": lambdas.append( lambda x: jitter(x, 256, 5))
        elif t == "jitter_256_1": lambdas.append( lambda x: jitter(x, 256, 1))
        else: raise Exception(f"Transform '{t}' not supported")
    
    return chain_lambdas(lambdas)

if __name__ == "__main__":
    import unittest

    l = [
        [3,4, -2],
        [2,5, -3]
    ]
    l = np.array(l)

    assert l.shape[0] == 2

    class Test_get_chained_transform(unittest.TestCase):
        def test_ordering(self):
            c1 = get_chained_transform(["times_two", "minus_two"])
            c2 = get_chained_transform(["minus_two", "times_two"])

            self.assertEqual(c1(2),0)
            self.assertEqual(c2(2),2)

    class Test_Chain_Lambdas(unittest.TestCase):
        def test_ordering(self):
            F = lambda x: x*2
            G = lambda x: x-2

            H1 = chain_lambdas([F,G])
            H2 = chain_lambdas([G,F])

            self.assertEqual(H1(2),0)
            self.assertEqual(H2(2),2)

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