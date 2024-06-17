import torch


def device_test(test_func):
    def wrapper():
        with torch.device('cpu'):
            test_func()
        with torch.device('meta'):
            test_func()
    return wrapper
