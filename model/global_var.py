import torch
# from odk.ApprxPICPyTorch import ApprxPICPyTorch as ApprSimu
from odk_main.ApprxPICPyTorchCuda import ApprxPICPyTorchCuda as ApprSimu
# from odk_main.ApprxPICPyTorch import ApprxPICPyTorch as ApprSimu
# from odk.cuda_0324 import ApprxPICCuda as ApprSimu
# from cuda4 import approxCuda as ApprSimu

global debug
debug = False 

global count_log
count_log = 0

use_compile = True

#instFolder = "/data/pace2/16X16_enob6.44"
#instFolder = "/data/pace2/16X16_enob6.09"
#instFolder= "/data/pace2/16X16_enob5.0_dacEnob6"
# instFolder= "/data/pace2/insts_enob5.38_1e-11"
#instFolder= "/data/pace2/insts_enob4.5_2e-11"
#instFolder= "/data/pace3/16X16_0401"
#instFolder= "/data/pace3/16X16_dacenob7.5_power0.025_noise2e-11_2gclock" # 20240422signoff
#instFolder= "/data/pace3/16X32_dacenob7.5_power0.05_noise2e-11_2g" # 20240422signoff
# instFolder= "/data/pace2/32X2_0403_180" # pace2
# instFolder = "/data/pace3/16X32_dacenob7.5_power0.07_noise1e11_2g" # std=37.93,mean=-0.0513 
#instFolder = "/data/pace3/16X16_dacenob7.5_power0.035_noise1e-11_2gclock" # std = 33.30, mean = 0.515 
# 20240506
# omac_config = {"instFolder": "/data/pace3/16X32_dacenob7.5_power0.05_noise1e-11_1gclock", "kdim": 16, "ndim": 32}
# omac_config = {"instFolder": "/data/pace3/16X32_dacenob7.5_power0.05_noise2e-11_1gclock", "kdim": 16, "ndim": 32}
# omac_config = {"instFolder": "/data/pace3/16X32_dacenob7.5_power0.07_noise1e-11_1gclock", "kdim": 16, "ndim": 32}
# omac_config = {"instFolder": "/data/pace3/16X32_dacenob7_power0.05_noise1e-11_1gclock", "kdim": 16, "ndim": 32}
# 20240508
# omac_config = {"instFolder": "/data/pace3/16X16_dacenob7.5_power0.035_noise1e-11_1gclock", "kdim": 16, "ndim": 16}
# omac_config = {"instFolder": "/data/pace3/16X16_dacenob7.5_power0.025_noise2e-11_1gclock", "kdim": 16, "ndim": 16}
# 20240513
omac_config = {"instFolder": "/data/pace3/16X16_dacenob7.5_power0.035_noise1e-11_500mhzclock", "kdim": 16, "ndim": 16}
#omac_config = {"instFolder": "/data/pace3/16X16_dacenob7.5_power0.025_noise2e-11_500mhzclock", "kdim": 16, "ndim": 16}
#omac_config = {"instFolder": "/data/pace3/16X32_dacenob7.5_power0.07_noise1e-11_500mhzclock", "kdim": 16, "ndim": 32}
#omac_config = {"instFolder": "/data/pace3/16X32_dacenob7.5_power0.05_noise2e-11_500mhzclock", "kdim": 16, "ndim": 32}


def init_ideal_simulator():
    from omac_1 import OPU
    global _simulator_instance 
    _simulator_instance = OPU(dev="cuda")
    print("Initialize a simulator at device:", _simulator_instance.device)
    if use_compile:
        _simulator_instance = torch.compile(_simulator_instance)

def init_multi_ideal_simulator():
    from omac_1 import OPU
    global _simulator_instance 
    _simulator_instance = OPU(dev=f"cuda:{torch.distributed.get_rank()}")
    print("Initialize a simulator at device:", _simulator_instance.device)

def init_simulator():
    global _simulator_instance
    instFolder = omac_config["instFolder"]
    k_dim = omac_config['kdim']
    n_dim = omac_config["ndim"]
    _simulator_instance = ApprSimu(
                                    device_type="cuda",
                                    physical_k_dimension=k_dim,
                                    physical_n_dimension=n_dim,
                                    k_dimension=k_dim,
                                    n_dimension=n_dim,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    inst_folder=instFolder,
                                    )
    print(f"init simu {instFolder} at device: {_simulator_instance.device_type}")
    if use_compile:
        _simulator_instance = torch.compile(_simulator_instance)

def init_multi_simulator():
    global _simulator_instance
    instFolder = omac_config["instFolder"]
    k_dim = omac_config['kdim']
    n_dim = omac_config["ndim"]
    _simulator_instance = ApprSimu(
                                    device_type=f"cuda:{torch.distributed.get_rank()}",
                                    physical_k_dimension=k_dim,
                                    physical_n_dimension=n_dim,
                                    k_dimension=k_dim,
                                    n_dimension=n_dim,
                                    input_precision=4,
                                    weight_precision=4,
                                    output_precision=8,
                                    inst_folder=instFolder,
                                    )
    print(f"init simu {instFolder} at device: {_simulator_instance.device_type}")
    if use_compile:
        _simulator_instance = torch.compile(_simulator_instance)
    if torch.distributed.get_rank() == 0:
        print(f" noise_mse: {_simulator_instance.noise_mse[:k_dim]}")
        print(f" gain: {_simulator_instance.gain[:k_dim]}")
        print(f" real_lsb: {_simulator_instance.real_lsb[:k_dim]}")
        print(f" offset_error: {_simulator_instance.offset_error[:k_dim]}")
        print(f" offset: {_simulator_instance.offset[:k_dim]}")
        print(f" scaling_factor: {_simulator_instance.scaling_factor[:k_dim]}")

