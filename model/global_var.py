import torch
from odk_story4.ApprxPICPyTorch import ApprxPICPyTorch as approxCuda
# from cuda3 import approxCuda
from omac_1 import OPU

instFolder = "/data/pace2/16X16_enob6.44"
# instFolder = "/data/pace2/16X16_enob5.0_dacEnob6"

def init_ideal_simulator():
    global _simulator_instance 
    _simulator_instance = OPU(dev="cuda")
    print("Initialize a simulator at device:", _simulator_instance.device)
    _simulator_instance = torch.compile(_simulator_instance)

def init_multi_ideal_simulator():
    global _simulator_instance 
    _simulator_instance = OPU(dev=f"cuda:{torch.distributed.get_rank()}")
    print("Initialize a simulator at device:", _simulator_instance.device)

def init_simulator():
    global _simulator_instance
    _simulator_instance = approxCuda(
                                    device_type="cuda",
                                    n_dimension_expendent = 16,
                                    k_dimension=16,
                                    n_dimension=16,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    instFolder=instFolder,
                                    )
    print(f"init simu {instFolder} at device: {_simulator_instance.device_type}")
    # _simulator_instance = torch.compile(_simulator_instance)

def init_multi_simulator():
    global _simulator_instance
    _simulator_instance = approxCuda(
                                    device_type=f"cuda:{torch.distributed.get_rank()}",
                                    n_dimension_expendent = 256,
                                    k_dimension=16,
                                    n_dimension=16,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    instFolder=instFolder,
                                    )
    print(f"init simu {instFolder} at device: {_simulator_instance.device_type}")
    _simulator_instance = torch.compile(_simulator_instance)
