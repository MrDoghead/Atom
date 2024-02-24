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
    out_bits= 12 # max output bits：log2(pic_row*144)*2 = log2(2304)*2 = 13bit
    tia_noise_sigma = 4  # std of pace noise 1 4 16
    tia_noise_mean = 0  # mean of pace noise
    max_out_bits = int(input_bits + weight_bits + np.log2(pic_row) )
    print(f"[OPU INFO]\n pic_row: {pic_row}\n pic_column:{pic_column}\n" \
            f" input_bits:{input_bits}\n weight_bits:{weight_bits}\n out_bits:{out_bits}\n" \
            f" tia_noise_sigma:{tia_noise_sigma}\n tia_noise_mean:{tia_noise_mean}")

    @staticmethod
    def forward(ctx, input,  weight):

        # max_out_bits = OPU.input_bits + OPU.weight_bits + np.log2(OPU.pic_row) 

        with torch.no_grad():
            out_mat = torch.matmul(input,weight)

            ## bitshift
            # mask = torch.where(out_mat>=0, 1, -1) 
            # out_mat_abs = torch.abs(out_mat)
            # output = out_mat_abs.type(torch.int32) >> (OPU.max_out_bits - OPU.out_bits)  #out_bits=8
            # out_mat = output*mask
            # out_mat = out_mat.type(torch.float32)

            # add TIA noise to output
            noise = torch.randn(size=out_mat.size(), requires_grad=False, device=input.device)*OPU.tia_noise_sigma + OPU.tia_noise_mean
            # out_mat += noise*(input_scale*weight_scale)
            out_mat += noise

            # round & clamp
            # out_mat = torch.round(out_mat)
            # low_bound =  -(2**(OPU.out_bits-1)) # -128
            # high_bound = 2**(OPU.out_bits-1) -1  #127
            # mask = (out_mat.ge(low_bound) * out_mat.le(high_bound))
            # out_mat = torch.clamp(out_mat, low_bound, high_bound) 
            # out_mat = torch.clamp(out_mat, low_bound, high_bound) + (2**(OPU.out_bits-1))

            # shift the integer back into the intended range
            # out_mat = out_mat * (1 << int(OPU.max_out_bits - OPU.out_bits))

            #converting backto floating point data
            # out_mat = out_mat_int

            return out_mat
