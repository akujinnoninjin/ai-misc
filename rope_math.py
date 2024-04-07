import math
import matplotlib.pyplot as plt

### Working through some visualisations of how RoPE changes the angle
### I think this is only linear rope. It might be NTK? It's *not* YaRN.

# For SOLAR:
    # dim = 4096
    # heads = 32
    # head size = 128
    # head dim = 0,127

    # freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);

    # period = rope_base ** (head_dim / head_size)
        # rope_base ** (0,1)
        #  1, rope_base
    # freq = 1,1/rope_base

# val = ctx position * freq
    # so care about 0-8k vs 8k+

# rope_base for 8k on a 4k model is 32k vs 10000


poss = [2048,6144] # [0, 2048, 4096, 6144, 8192]
rope_bases = [10000, 32000]
head_size = 128
head_dims = [0,32,64,96,128]

plot_data = {}

for rope_base in rope_bases:
    plot_data[rope_base] = {}

    for head_dim in head_dims:
        plot_data[rope_base][head_dim] = {'sin':[], 'cos':[]}

        for pos in (range(0,4096) if rope_base == 10000 else range(0,8192)): # poss:
            period = rope_base ** (head_dim / head_size)
            freq = 1/period
            val = pos * freq
            plot_data[rope_base][head_dim]['sin'].append(math.sin(val))
            plot_data[rope_base][head_dim]['cos'].append(math.cos(val))
 

print(period)
            #fci.append(math.sin(val))
# plt.plot(range(0,8192,2),plot_data[10000][64]['sin'])
# plt.plot(range(0,4096),plot_data[10000][64]['sin'])
# plt.plot(range(0,8192),plot_data[32000][64]['sin'])
# # plt.plot(range(0,8192),plot_data[32000][64]['cos'])
# plt.show()