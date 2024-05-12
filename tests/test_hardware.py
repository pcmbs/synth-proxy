from urllib.request import urlopen
import pytest
import torch


def test_gpu_available():
    assert torch.cuda.is_available()
