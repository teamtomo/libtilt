import torch


def device_test(test_func):
    def decorator():
        with torch.device('cpu'):
            test_func()
        with torch.device('cuda'):
            test_func()
    return decorator
