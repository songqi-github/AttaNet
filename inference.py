
import os
import time
import argparse

import torch
import torch.nn.functional as F

from AttaNet_light import AttaNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# args
parse = argparse.ArgumentParser()
parse.add_argument('--ckpt', dest='ckpt', type=str, default='./snapshots/AttaNet_light_706.pth',)
args = parse.parse_args()

# define model
net = AttaNet(n_classes=19)
net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
net.cuda()
net.eval()

input_t = torch.Tensor(1, 3, 1024, 2048).cuda()

print("start warm up")
for i in range(10):
    net(input_t)
print("warm up done")

start_ts = time.time()
for i in range(500):
    input = F.interpolate(input_t, (512, 1024), mode='nearest')
    net(input)
end_ts = time.time()

t_cnt = end_ts - start_ts
print("=======================================")
print("FPS: %f" % (500 / t_cnt))
print("Inference time %f ms" % (t_cnt/500*1000))
