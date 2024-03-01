import torch
from odk_story4.ApprxPICPyTorch import ApprxPICPyTorch

def init_simulator():
    global _simulator_instance
    _simulator_instance = ApprxPICPyTorch(
                                    device_type="cuda",
                                    n_dimension_expendent = 256,
                                    k_dimension=16,
                                    n_dimension=16,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    instFolder="/data/pace2/16X16_0",
                                    )
    print("Initialize a simulator at device:", _simulator_instance.device_type)
    _simulator_instance = torch.compile(_simulator_instance)

def init_multi_simulator():
    global _simulator_instance
    _simulator_instance = ApprxPICPyTorch(
                                    device_type=f"cuda:{torch.distributed.get_rank()}",
                                    n_dimension_expendent = 256,
                                    k_dimension=16,
                                    n_dimension=16,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    instFolder="/data/pace2/16X16_0",
                                    )
    print("Initialize a simulator at device:", _simulator_instance.device_type)
    _simulator_instance = torch.compile(_simulator_instance)
