# from __future__ import print_function
import sys
# To avoid error if ROS was installed in the device
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import cv2
import os
import sys
import argparse
import traceback
import glob
import h5py
import time
import numpy as np
import multiprocessing 
from dataset.config import BF_CONFIG
import utils.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def sendmail_to_bigjun(error):
    pass

def video2images(path):
    # prepare directory
    if args.ds == "breakfast":
        _, _, _, _, p1, p2, p3= path.split("/")
        out_dir = os.path.join(args.out_dir, p1, p2, p3.split("_")[1].split(".")[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # reading video
    capture = cv2.VideoCapture(path)
    # check whether sucessfully open the video or not
    if not capture.isOpened():
        print("Failed to open the video!!!")
        sys.exit(1)

    # get basic information of the video
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fcount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fwidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    que.put(round(fcount/fps/60, 1))
    print("fps=%d fcount=%d fwidth=%d fhigjt=%d" %(fps, fcount, fwidth, fheight))

    # extract the frames from the video
    EXTRACT_FREQUENCY = BF_CONFIG['EXTRACT_FREQUENCY']
    RESIZE_WIDTH = BF_CONFIG['RESIZE_WIDTH']
    RESIZE_HEIGHT = BF_CONFIG['RESIZE_HEIGHT']
    count = 0
    for i in range(fcount):
        ret, frame = capture.read()
        if not ret and frame is None:
            print("Miss a frame!!!")
            continue
        # save frame with specified frequency
        if count % EXTRACT_FREQUENCY == 0:
            if (RESIZE_HEIGHT != fheight) or (RESIZE_WIDTH != fwidth):
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            cv2.imwrite(filename='{}/{:06d}.jpg'.format(out_dir, i+1), img=frame)
        count += 1

    # release the videoCapture object once it is no longer needed
    capture.release()

def video2hdf5(path):
    # prepare directory
    if args.ds == "breakfast":
        _, _, _, _, p1, p2, p3= path.split("/")
        out_dir = os.path.join(args.out_dir, p1, p2, p3.split("_")[1].split(".")[0])

    # reading video
    capture = cv2.VideoCapture(path)
    # check whether sucessfully open the video or not
    if not capture.isOpened():
        print("Failed to open the video!!!")
        sys.exit(1)

    # get basic information of the video
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fcount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fwidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_video_len.append(round(fcount/fps/60, 1))
    print("fps=%d fcount=%d fwidth=%d fhigjt=%d" %(fps, fcount, fwidth, fheight))

    # extract the frames from the video
    EXTRACT_FREQUENCY = BF_CONFIG['EXTRACT_FREQUENCY']
    RESIZE_WIDTH = BF_CONFIG['RESIZE_WIDTH']
    RESIZE_HEIGHT = BF_CONFIG['RESIZE_HEIGHT']

    count = 0
    all_frames = None
    for i in range(fcount):
        ret, frame = capture.read()
        if not ret and frame is None:
            continue
        # save frame with specified frequency
        if count % EXTRACT_FREQUENCY == 0:
            try:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                all_frames = np.concatenate((all_frames, frame[None,:])) if isinstance(all_frames, np.ndarray) else frame[None,:]
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                sendmail_to_bigjun(error)
                print(error)
        count += 1
    breakfast_raw_img.create_dataset(name=out_dir, data=all_frames)
    # import ipdb; ipdb.set_trace()
    # release the videoCapture object once it is no longer needed
    capture.release()    

def plot_all_video_len(data):
    x = range(1, len(data)+1)
    plt.plot(x, data, 'ro-', ms=3)
    plt.ylabel('duration /min')
    plt.xlabel('index')
    plt.savefig(args.out_dir.replace('rgb_frame/' ,"all_video_len.jpg"))
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='convert video into frames')
    parser.add_argument('src_dir', type=str, help="source video directory")
    parser.add_argument('out_dir', type=str, help="output frames directory")
    parser.add_argument("--ds", "--dataset", type=str, default='breakfast', choices=['breakfast', 'charades'], help="which dataset do you want to process")
    parser.add_argument("--nw", "--num_workers", type=int, default=1, help="number of workers to extract rawframes")
    parser.add_argument("-m", "--mode", type=str, default='frame', choices=['frame', 'hdf5'], help='Transform video into which mode [frame or hdf5]')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # get the list of video in corresponding dataset
    if args.ds == 'breakfast':
        video_list = glob.glob(args.src_dir+"*/*/*")
        video_list = [i for i in video_list if i.split('.')[-1] != "labels" and 'ch1' not in i]
    elif args.ds == "charades":
        video_list = glob.glob(args.src_dir+"*/*/*")
    print(f"The number of video in {args.ds} is:", len(video_list))
    
    t1 = time.time()
    if args.mode == 'frame':
        que = multiprocessing.Queue()
        pool = multiprocessing.Pool(args.nw)
        pool.map(video2images, video_list)
        all_video_len = []
        for _ in range(que.qsize()):
            all_video_len.append(que.get())
        io.dumps_json(all_video_len, args.out_dir.replace('rgb_frame/' ,"all_video_len.json"))
        # all_video_len = io.loads_json(args.out_dir.replace('rgb_frame/' ,"all_video_len.json"))
        plot_all_video_len(sorted(all_video_len))
    # WARNING: It will take much space to use h5py to save the raw date    
    else:
        breakfast_raw_img = h5py.File(os.path.join(args.out_dir, 'raw_img_data.hdf5'), 'w')
        all_video_len = []
        for i in video_list:
            video2hdf5(i)
        breakfast_raw_img.close()
        io.dumps_json(all_video_len, args.out_dir.replace('rgb_frame/' ,"all_video_len.json"))
        plot_all_video_len(sorted(all_video_len))        
    t2 = time.time()
    print("Finished processing {} videos.".format(len(video_list)), f"It takes {t2-t1} seconds!!!")