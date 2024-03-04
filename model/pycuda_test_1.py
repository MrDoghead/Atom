#https://blog.csdn.net/BigerBang/article/details/106981253
#https://zhuanlan.zhihu.com/p/645052868?utm_id=0
import numpy as np
import torch
# import pycuda.autoinit
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
# import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

vmap = np.load('/data/omacshit/vmap.npy').astype(np.float32)
wmap = np.load('/data/omacshit/wmap.npy')
wmap = np.mean(wmap,axis=1).astype(np.float32)
# print(vmap.shape, wmap.shape)
omac_size =16

# for i in range(100):
#     a = torch.randn((1024,1024)).cuda()
#     torch.matmul(a, a.T)

def _map_v(inp,arr):
    out = torch.zeros_like(inp,dtype=torch.float)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            for k in range(inp.shape[2]):
                out[i][j][k]=arr[k][inp[i][j][k]]
    return out

def _map_w(inp,arr):
    out = torch.zeros_like(inp,dtype=torch.float)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            for k in range(inp.shape[2]):
                out[i][j][k]=arr[j][inp[i][j][k]]
    return out

# x = torch.cuda.FloatTensor(8) #不加这个就报错:pycuda._driver.LogicError: cuFuncSetBlockShape failed: invalid resource handle

class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder,self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    
    def get_point(self):
        return self.t.data_ptr()

# inputs should be int32/fp32
mod = SourceModule("""
void __global__  mapping(const int* input, const int *input_dims, float* input_out, const float *corrected_v_values,
                         const int* weight, const int *weight_dims, float* weight_out, const float *corrected_w_values)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int input_lenght = input_dims[0] * input_dims[1] * input_dims[2];
    int weight_lenght = weight_dims[0] * weight_dims[1] * weight_dims[2];
    int lenght = max(input_lenght,weight_lenght);
    int omac_size = input_dims[2];

    for (int idx = index; idx < lenght; idx += stride)
    {   
        
        if(idx<input_lenght){
            int i_v = input[idx];
            int ch = idx % omac_size;
            input_out[idx] = corrected_v_values[omac_size * ch + i_v];
        }

        if(idx<weight_lenght){
            int i_w = weight[idx];
            int blk = idx % (weight_dims[1]*weight_dims[2]);
            int ch = blk / weight_dims[2];
            weight_out[idx] = corrected_w_values[omac_size * ch + i_w];            
        }
    }
   
}
"""
)
mapping = mod.get_function("mapping")

mod2 = SourceModule("""
void __global__  mapping(const int* input, const int *input_dims, float* input_out, const float *corrected_v_values,
                         const int* weight, const int *weight_dims, float* weight_out, const float *corrected_w_values)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    int input_length = input_dims[0] * input_dims[1] * input_dims[2];
    int weight_length = weight_dims[0] * weight_dims[1] * weight_dims[2];
    int max_length = max(input_length, weight_length);
    int omac_size = input_dims[2];
                   
    __shared__ float s_vmap[256];
    __shared__ float s_wmap[256];
    if (index < max_length){
        int sid = index % 256;
        s_vmap[sid] = corrected_v_values[sid];
        s_wmap[sid] = corrected_w_values[sid];
    }
    __syncthreads();

    for (int idx = index; idx < max_length; idx += stride)
    {   
        
        if(idx<input_length){
            int i_v = input[idx];
            int ch = idx % omac_size;
            input_out[idx] = s_vmap[omac_size * ch + i_v];
        }

        if(idx<weight_length){
            int i_w = weight[idx];
            int blk = idx % (weight_dims[1]*weight_dims[2]);
            int ch = blk / weight_dims[2];
            weight_out[idx] = s_wmap[omac_size * ch + i_w];            
        }
    }   
}
"""
)
mapping2 = mod2.get_function("mapping")

def test():

    input = torch.randint(low=0, high=16, size=(8, 2048, omac_size),dtype=torch.int32, device='cuda')  #[256,116,16]
    weight = torch.randint(low=-8, high=8, size=(8, omac_size, 4096),dtype=torch.int32, device='cuda') #[256,16,4096]

    input_out = torch.zeros(input.shape,dtype=torch.float, device='cuda')
    input_dims = torch.tensor(input.shape,dtype=torch.int32, device='cuda')
    corrected_v_values = torch.from_numpy(vmap).to('cuda')

    weight = weight+8
    weight_out = torch.zeros(weight.shape,dtype=torch.float, device='cuda')  ##
    weight_dims = torch.tensor(weight.shape,dtype=torch.int32, device='cuda')
    corrected_w_values = torch.from_numpy(wmap).to('cuda')

    # 将sequence lenght变成32的整数倍
    # sq = int(np.ceil(input.shape[1]/32)*32)
    blocksPerGrid = max(input.shape[0], weight.shape[0])
    threadsPerBlock = 256
    start = drv.Event()
    end=drv.Event()
    start.record()
    mapping2(
            Holder(input), Holder(input_dims), Holder(input_out), Holder(corrected_v_values),
            Holder(weight), Holder(weight_dims), Holder(weight_out), Holder(corrected_w_values),
            grid=(blocksPerGrid,1,1), block=(threadsPerBlock,1,1))
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("pycuda time cost:", secs)

    import time
    st = time.perf_counter()
    input_out_ = _map_v(input,corrected_v_values)
    weight_out_ = _map_w(weight,corrected_w_values)
    et = time.perf_counter()
    print("loop time cost:", et-st)

    print(torch.allclose(input_out.type(torch.float), input_out_))
    print(torch.allclose(weight_out.type(torch.float), weight_out_))

test()