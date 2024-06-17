import torch
import functools


def device_test(test_func):
    @functools.wraps(test_func)
    def run_devices(*args, **kwargs):
        for device in ('cpu', 'cuda'):
            with torch.device(device):
                test_func(*args, **kwargs)
    return run_devices
