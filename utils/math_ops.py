import numpy as np


def is_in_range(x_input, ref_range):
    return ref_range[0] <= x_input <= ref_range[1]


def scale_in_range(x_input, ref_range, scale_factor):
    return x_input + scale_factor * (ref_range[1] - ref_range[0])


def scale_max_to(vals, ext_max):
    res = [v * ext_max / max(vals) for v in vals]
    return res


def gaussian(x, amplitude, mean, sigma):
    return np.float(amplitude)*(1/(np.float(sigma)*(np.sqrt(2*np.pi))))*(np.exp(-((x-np.float(mean))**2)/((2*sigma)**2)))


def get_bin_centers(values):
    return [(values[i] + values[i + 1]) * 0.5 for i in range(len(values) - 1)]
