import torch
import numpy as np
import torch.nn.functional as F
from pycuda_test_1 import Holder, mapping

class OPU(torch.nn.Module):
    """
    Optical Processing Unit (OPU) Design
 
    """
    def __init__(self, dev):
        super().__init__()
        self.device = dev
        self.pic_row = 16 # 必须是2的整数次方
        self.pic_column = 16 # 必须是2的整数次方
        self.input_bits = 4
        self.weight_bits = 4
        self.adc_output_bits = 8 #
        self.tia_noise_sigma = 0 #1.15  # fixed value, indepenet from tia gain @Yuemiao Di
        self.tia_noise_mean = 0  #
        self.tia_gain = 1 # [1 2 4 8 16],  defalut value is 1, choose the best one
        self.base_scale_factor = 2**int(self.input_bits + self.weight_bits + np.log2(self.pic_row) - self.adc_output_bits)
        self.vmap = torch.from_numpy(np.load("/data/omacshit/vmap.npy")).to(torch.float32).to(self.device)
        self.wmap = torch.from_numpy(np.load("/data/omacshit/wmap.npy")).mean(dim=1).to(torch.float32).to(self.device)
        print(f"[OPU INFO]\n pic_row: {self.pic_row}\n pic_column:{self.pic_column}\n" \
            f" input_bits:{self.input_bits}\n weight_bits:{self.weight_bits}\n adc_output_bits:{self.adc_output_bits}\n" \
            f" tia_noise_sigma:{self.tia_noise_sigma}\n tia_noise_mean:{self.tia_noise_mean}\n tia_gain:{self.tia_gain}\n" \
            f" base_scale_factor:{self.base_scale_factor}")
     
    @torch.no_grad
    def optical_matmul(self, input, weight):
        def omm(v, w):
            mm_out = torch.matmul(v, w)

            scale_factor = self.base_scale_factor / self.tia_gain

            mm_out = mm_out / scale_factor

            # add TIA noise to 8bit output
            tia_noise = torch.normal(mean=self.tia_noise_mean, std=self.tia_noise_sigma, size=mm_out.size(),  device=input.device)
                
            adc_out = torch.clamp(torch.round(mm_out + tia_noise),
                                  -2**(self.adc_output_bits-1), 2**(self.adc_output_bits-1)-1)

            scaled_out = adc_out * scale_factor # scale adc output back to intended range in digital domain

            return scaled_out

        input_ = input + 8
        input_offset = torch.zeros(input_.shape, dtype=torch.float32, device=input_.device)
        input_dims = torch.tensor(input_.shape, dtype=torch.int32, device=input_.device)

        weight_ = weight + 8 # for indexing
        weight_offset = torch.zeros(weight_.shape, dtype=torch.float32, device=weight_.device)
        weight_dims = torch.tensor(weight_.shape, dtype=torch.int32, device=weight_.device)

        grid_size = max(input_.shape[0], weight_.shape[0])
        block_size = self.pic_row * self.pic_column
        mapping(
            Holder(input_.to(torch.int32)), Holder(input_dims), Holder(input_offset), Holder(self.vmap),
            Holder(weight_.to(torch.int32)), Holder(weight_dims), Holder(weight_offset), Holder(self.wmap),
            grid=(grid_size, 1, 1), block=(block_size, 1, 1))

        input_out = input + input_offset
        weight_out = weight + weight_offset
        omm_out = omm(input_out, weight_out)

        return omm_out
    
    def forward(self, input, weight):
        input_dims = len(input.shape)
        assert(input.shape[-1] == weight.shape[0])
        batch_size = input.shape[0]
        repeats = input.shape[-1] // self.pic_row
        remainder = input.shape[-1] % self.pic_row
        repeats = repeats+1 if remainder!=0 else repeats
        
        pad_dim = self.pic_row*repeats-input.shape[-1]

        if(input_dims==3):
            inp_ = F.pad(input,(0, pad_dim, 0, 0),"constant",value=0)#左右上下,填右边
            inp_ = inp_.contiguous().view([batch_size*input.shape[1], repeats, -1])
            # input_scale_ = F.pad(input_scale,(0, pad_dim, 0, 0),"constant",value=0)
            # input_scale_ = input_scale_.contiguous().view([batch_size*input.shape[1], repeats, -1])
        elif (input_dims==2):
            inp_ = F.pad(input,(0, pad_dim, 0, 0),"constant",value=0)#左右上下,填右边
            inp_ = inp_.contiguous().view(batch_size, repeats, -1)
            # input_scale_ = F.pad(input_scale,(0, pad_dim, 0, 0),"constant",value=0)
            # input_scale_ = input_scale_.contiguous().view(batch_size, repeats, -1)
        else:
            raise NotImplementedError
        
        weight_ = F.pad(weight, (0, 0, 0, pad_dim), "constant", value=0)#左右上下， 填下边
        weight_ = weight_.permute(1,0).reshape(weight_.shape[1], repeats, -1).permute(2,1,0)
        # weight_scale_ = F.pad(weight_scale, (0, 0, 0, pad_dim), "constant", value=0)#左右上下， 填下边
        # weight_scale_ = weight_scale_.permute(1,0).reshape(weight_scale_.shape[1], repeats, -1).permute(2,1,0)

        weight_t = weight_.permute(1,0,2) 
        # weight_scale_t = weight_scale_.permute(1,0,2)

        input_t = inp_.permute(1,0,2)
        # input_scale_t = input_scale_.permute(1,0,2)
        
        # temp_out_true = torch.matmul(input_t, weight_t)
        temp_out = self.optical_matmul(input_t, weight_t)
        temp_out = temp_out.permute(1,0,2)
        temp_out = torch.sum(temp_out, dim=1, keepdim=False)


        if len(input.shape)==2:
            return temp_out
        elif len(input.shape)==3:
            out_ = temp_out.view([batch_size,input.shape[1],weight.shape[1]]) 
            return out_
            
