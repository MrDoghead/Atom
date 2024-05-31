import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

colors = ['red', 'orangered', 'yellow', 'green', 'cyan', 'blue', 'purple']
matrices = ['q', 'k', 'v', 'o', 'g', 'd', 'u']

def process_gptq(file_path):
    raw_data = np.zeros((7, 32), dtype=np.float16) # llama2
    with open(file_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            items = line.strip().split('\t')
            assert len(items) == 3, f"line: {i}"
            name = items[0].split()[-1]
            error = float(items[1].split()[-1])
            time = float(items[2].split()[-1])
            layer_id = i // 7
            weight_id = i % 7
            raw_data[weight_id][layer_id] = error
    # draw line graph
    x = np.arange(32)
    for i in range(7):
        y = raw_data[i].flatten()
        plt.plot(x, y, color = colors[i], label=matrices[i])
    output_name = "llama2-gptq-error-w4-g16"
    plt.xlabel("layers")
    plt.ylabel("error")
    plt.title(output_name)
    plt.legend()
    plt.savefig(f"{output_name}.png")


class LlamaOmacDist:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.n_proj = 7
        self.dist = {}
        for i in range(n_layers):
            self.dist[i] = {
                "q_proj":    [Counter(), Counter()],
                "k_proj":    [Counter(), Counter()],
                "v_proj":    [Counter(), Counter()],
                "o_proj":    [Counter(), Counter()],
                "gate_proj": [Counter(), Counter()],
                "up_proj":   [Counter(), Counter()],
                "down_proj": [Counter(), Counter()],
            }
        self.module_map = {
                0: "q_proj",
                1: "k_proj",
                2: "v_proj",
                3: "o_proj",
                4: "gate_proj",
                5: "up_proj",
                6: "down_proj"
            }

    def plot_dist(self, n_layers):
        n_proj = self.n_proj
        plt.figure(figsize=(n_proj*2*5, n_layers*5))
        for i in range(n_layers):
            for j in range(n_proj):
                idx = i * n_proj * 2 + j * 2 + 1
                # plot inputs
                items1, counts1 = zip(*self.dist[i][self.module_map[j]][0].items())
                plt.subplot(n_layers, n_proj*2, idx)
                plt.bar(items1, counts1, color='blue')
                plt.title(f'layer-{i} {self.module_map[j]} input')
                plt.xlabel('Val')
                plt.ylabel('Freq')

                # plot weights
                items2, counts2 = zip(*self.dist[i][self.module_map[j]][1].items())
                plt.subplot(n_layers, n_proj*2, idx+1)
                plt.bar(items2, counts2, color='red')
                plt.title(f'layer-{i} {self.module_map[j]} weight')
                plt.xlabel('Val')
                plt.ylabel('Freq')

        # plt.tight_layout()
        # plt.show()
        output_name = "llama2-omac-input-dist"
        plt.savefig(f"{output_name}.png")
    
# Must follow the format:
# DecodeLayer-0
# `module` `x_proj`
# `block1`
# `block2`
# ...
# ......
# DecodeLayer-1
# .........
#
# `module` = attn | mlp
# `x_proj` = q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj
# `block`  = {
# input shape0 shape1
# ...matrix-by-row...
# weight shape0 shape1
# ...matrix-by-row...
# }
def omac_dist(log_file, first_n_layers=0):
    dist = LlamaOmacDist(n_layers=32)
    line_idx = 0
    block_idx = 0
    decode_layer_idx = 0
    fill_idx = 0
    x = None
    w = None
    timestamp = 0
    with open(log_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line_idx += 1
            if "DecodeLayer" in line: # e.g. DecodeLayer-0
                item = line.strip().split('-')
                decode_layer_idx = int(item[1])
                if timestamp:
                    old_timestamp = timestamp
                    timestamp = time.perf_counter()
                    print(f"* time cost: {timestamp-old_timestamp} s")
                else:
                    timestamp = time.perf_counter()
                if decode_layer_idx == first_n_layers:
                    break
                print(f"Start reading DecodeLayer-{decode_layer_idx}")
                continue
            if "attn" in line or "mlp" in line: # e.g. attn q_proj
                item = line.strip().split()
                proj = item[1]
                block_idx = 0   # start count blocks
                x = None
                w = None
                x_shape0, x_shape1 = 0, 0
                w_shape0, w_shape1 = 0, 0
                continue
            if "input" in line:
                item = line.strip().split()
                x_shape0, x_shape1 = int(item[1]), int(item[2])
                x = Counter()
                fill_idx = 0
                block_idx += 1
                continue
            if "weight" in line:
                item = line.strip().split()
                w_shape0, w_shape1 = int(item[1]), int(item[2])
                w = Counter()
                fill_idx = 0
                block_idx += 1
                continue
            if isinstance(x, Counter):
                item = line.strip().split()
                arr = [int(k) for k in item]
                x.update(Counter(arr))
                fill_idx += 1
                if fill_idx == x_shape0:
                    dist.dist[decode_layer_idx][proj][0].update(Counter(x))
            if isinstance(w, Counter):
                item = line.strip().split()
                arr = [int(k) for k in item]
                w.update(Counter(arr))
                fill_idx += 1
                if fill_idx == w_shape0:
                    dist.dist[decode_layer_idx][proj][1].update(Counter(w))
    dist.plot_dist(n_layers=first_n_layers)


if __name__ == "__main__":
    # process_gptq(file_path="./logs/gptq-w4-g16.log")
    omac_dist(log_file="./logs/omac_16X16_dacenob7.5_power0.035_noise1e-11_500mhzclock.bkp.txt", first_n_layers=8)
    # omac_dist(log_file="./logs/tmp")