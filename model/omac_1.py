import torch
import numpy as np

class OPU(torch.autograd.Function):
    """ 
    Optical Processing Unit (OPU) Design 

    """
    pic_row= 16 # 必须是2的整数次方
    pic_column= 16 # 必须是2的整数次方
    input_bits= 4
    weight_bits= 4
    out_bits= 8 # max output bits：log2(pic_row*144)*2 = log2(2304)*2 = 13bit
    tia_noise_sigma = 2  # std of pace noise 1 4 6
    tia_noise_mean = 0  # mean of pace noise
    tia_scaling_factor = 16 # 2 4 8 16
    max_out_bits = int(input_bits + weight_bits + np.log2(pic_row) )
    print(f"[OPU INFO]\n pic_row: {pic_row}\n pic_column:{pic_column}\n" \
            f" input_bits:{input_bits}\n weight_bits:{weight_bits}\n out_bits:{out_bits}\n" \
            f" tia_noise_sigma:{tia_noise_sigma}\n tia_noise_mean:{tia_noise_mean}\n tia_scaling_factor:{tia_scaling_factor}")

    @staticmethod
    def forward(ctx, input,  weight):

        # max_out_bits = OPU.input_bits + OPU.weight_bits + np.log2(OPU.pic_row) 

        with torch.no_grad():
            mm_out = torch.matmul(input, weight) # signal 2 4 8 16

            # add TIA noise to output
            noise = torch.randn(size=mm_out.size(), requires_grad=False, device=input.device) * OPU.tia_noise_sigma + OPU.tia_noise_mean
            # omac_mm_out = mm_out + noise * 16
            # noise = torch.normal(mean=OPU.tia_noise_mean*(2**(OPU.max_out_bits - OPU.out_bits)), 
            #              std=OPU.tia_noise_sigma*(2**(OPU.max_out_bits - OPU.out_bits)), 
            #              size =mm_out.size(),  device=input.device)
            # omac_mm_out = mm_out + noise
            omac_mm_out = mm_out

            # scale
            omac_mm_out = omac_mm_out * OPU.tia_scaling_factor
            mask = torch.where(omac_mm_out>=0, 1, -1) 
            out_mat_abs = torch.abs(omac_mm_out)
            omac_mm_out = out_mat_abs.to(torch.int32) >> (OPU.max_out_bits - OPU.out_bits)
            omac_mm_out = omac_mm_out * mask
            omac_mm_out = omac_mm_out.clip(-128, 127)
            
            omac_mm_out = omac_mm_out + noise

            # recover
            omac_mm_out = omac_mm_out.to(torch.int32) << (OPU.max_out_bits - OPU.out_bits)
            omac_mm_out = omac_mm_out // OPU.tia_scaling_factor


            return omac_mm_out.to(torch.float32)
