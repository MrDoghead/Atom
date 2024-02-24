import torch
import torch.nn.functional as F
from omac_1 import OPU as opu

#自定义gemm操作
def GEMM_OPU_Parallel(input, weight): # 
    input_dims = len(input.shape)
    assert(input.shape[-1] == weight.shape[0])
    batch_size = input.shape[0]
    repeats = input.shape[-1] // opu.pic_row
    remainder = input.shape[-1] % opu.pic_row
    repeats = repeats+1 if remainder!=0 else repeats
    
    pad_dim=opu.pic_row*repeats-input.shape[-1]

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
    temp_out = opu.apply(input_t, weight_t)
    temp_out = temp_out.permute(1,0,2)
    temp_out = torch.sum(temp_out, dim=1, keepdim=False)


    if len(input.shape)==2:
        return temp_out
    elif len(input.shape)==3:
        out_ = temp_out.view([batch_size,input.shape[1],weight.shape[1]]) 
        return out_

# def GEMM_OPU_Parallel(input, weight): # 
#     input_dims = len(input.shape)
#     assert(input.shape[-1] == weight.shape[-2])
#     batch_size = input.shape[0]
#     repeats = input.shape[-1] // opu.pic_row
#     remainder = input.shape[-1] % opu.pic_row
#     repeats = repeats+1 if remainder!=0 else repeats
    
#     pad_dim=opu.pic_row*repeats-input.shape[-1]

#     if(input_dims==3):
#         inp_ = F.pad(input,(0, pad_dim, 0, 0),"constant",value=0)#左右上下,填右边
#         inp_ = inp_.contiguous().view([batch_size*input.shape[1], repeats, -1])
#     elif (input_dims==2):
#         inp_ = F.pad(input,(0, pad_dim, 0, 0),"constant",value=0)#左右上下,填右边
#         inp_ = inp_.contiguous().view(batch_size, repeats, -1)
#     else:
#         raise NotImplementedError
    
#     weight_ = F.pad(weight, (0, 0, 0, pad_dim), "constant", value=0)#左右上下， 填下边
#     weight_ = weight_.permute(1,0).reshape(weight_.shape[1], repeats, -1).permute(2,1,0)

#     weight_t = weight_.permute(1,0,2) 
#     inp_t = inp_.permute(1,0,2)
#     temp_out = torch.matmul(inp_t, weight_t)
#     # temp_out = opu.apply(inp_t, weight_t)
#     temp_out = temp_out.permute(1,0,2)
#     temp_out = torch.sum(temp_out, dim=1, keepdim=False)


#     if len(input.shape)==2:
#         return temp_out
#     elif len(input.shape)==3:
#         out_ = temp_out.view([batch_size,input.shape[1],weight.shape[1]]) 
#         return out_