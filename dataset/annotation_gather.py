import os
import time
import glob
import argparse
import collections
from multiprocessing import Pool, Queue
import utils.io as io

# main function to gather notation
def gather(f_name):
    f = open(f_name)
    lines = f.readlines()
    # anno = collections.OrderedDict()
    anno = {}
    for line in lines:
        line = line.rstrip()
        dur_str, act = line.split(" ")
        anno[dur_str] = act
        if act not in action_class:
            action_class.append(act)
    # anno = sorted(anno.items(), key=lambda d: int(d[0].split("-")[0]))k
    f_name = f_name.replace(f_name.split('/')[-1], f_name.split('/')[-1].split('_')[1].split('.')[0])
    f_name = f_name.replace('original', 'rgb_frame')
    if not os.path.exists(f_name):
        miss_video.append(f_name)
    # que.put({f_name: anno})
    labels[f_name] = anno

# set optional arguments
parser = argparse.ArgumentParser(description='gather all notation into single file.')
parser = argparse.ArgumentParser(description='convert video into frames')
parser.add_argument('src_dir', type=str,  help="source video directory")
parser.add_argument('out_dir', type=str,  help="output frames directory")
parser.add_argument("--ds", "--dataset", type=str, default='breakfast', choices=['breakfast', 'charades'], help="which dataset do you want to process")
parser.add_argument("--nw", "--num_workers", type=int, default=1, help="number of workers to extract rawframes")
args = parser.parse_args()

# get the list of notation file in corresponding dataset
if args.ds == 'breakfast':
    notation_list = glob.glob(args.src_dir+"*/*/*")
    notation_list = [i for i in notation_list if i.split('.')[-1] != "avi" ]
    print("{} files in total to be processed.".format(len(notation_list)))
elif args.ds == "charades":
    # TODO
    notation_list = glob.glob(args.src_dir+"*/*/*")

# processing
labels = {}
action_class = []
miss_video = []
t1 = time.time()
for f in notation_list:
    gather(f)

io.dumps_json(labels, os.path.join(args.out_dir, "notation.json"))
io.write(os.path.join(args.out_dir, "action_class.txt"), action_class, 'w')
io.write(os.path.join(args.out_dir, "miss_video.txt"), miss_video, 'a')
t2 = time.time()
print(f"Finished processing {len(labels.keys())}.", f"It takes {t2-t1} seconds!!!")

# # Note that the processes are independent with each other, we need "Queue()" to share space 
# que = Queue()
# pool = Pool(args.nw)
# pool.map(gather, notation_list)
# pool.close()
# pool.join()
# import ipdb; ipdb.set_trace()
# for _ in range(que.qsize()):
#     val = que.get()
#     for k, v in val.items():
#         labels[k] = v