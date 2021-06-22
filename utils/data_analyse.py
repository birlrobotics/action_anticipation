import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import h5py

from PIL import Image, ImageEnhance
from sklearn.manifold import TSNE
from dataset.config import BF_ACTION_CLASS, BF_ACTION_COLOR

ORI_TEST_DATA = None
NOR_TEST_DATA = None

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def vis_distribution(data, title='test', legend='none'):
    plt.title(title)
    for i, d in enumerate(data):
        sns.distplot(d, bins=50, kde=True)
    plt.legend(legend)
    plt.ion()
    plt.pause(5)
    plt.cla()

def vis_i3ddata_distribution():
    data_feat = h5py.File(os.path.join(BF_CONFIG["data_dir"], BF_CONFIG["feat_hdf5_name"]), 'r')
    data_nota = io.loads_json(os.path.join(BF_CONFIG["data_dir"], 'notation.json'))
    for dir in data_nota.keys():
        feat = data_feat[dir]['avg_feat'][:]
        label = data_feat[dir]['label'][:]
        for i in range(feat.shape[0]):
            data = pd.Series(feat[i])
            vis_distribution([data], dir, [BF_ACTION_CLASS[label[i]]])

def vis_i3ddata_tsne():
    data_feat = h5py.File(os.path.join(BF_CONFIG["data_dir"], BF_CONFIG["feat_hdf5_name"]), 'r')
    data_nota = io.loads_json(os.path.join(BF_CONFIG["data_dir"], 'notation.json'))
    all_data, all_label = None, None
    for dir in data_nota.keys():
        feat = data_feat[dir]['avg_feat'][:]
        label = data_feat[dir]['label'][:]
        all_data = np.concatenate((all_data, feat)) if all_data is not None else feat
        all_label = np.concatenate((all_label, label)) if all_label is not None else label
    
    e_idx = 1000
    x_tsne = TSNE(n_components=2, init='pca', random_state=33, perplexity=30, n_iter=1000).fit_transform(all_data[:e_idx, :])

    plt.figure(figsize=(10, 10))
    handles_list = []; labels_list = []
    for i, label in enumerate(all_label[:e_idx]):
        a_class = BF_ACTION_CLASS[label]
        globals()['p'+str(label)] = plt.scatter(x_tsne[i,0], x_tsne[i, 1], c=np.array(BF_ACTION_COLOR[a_class])/255.0, label=a_class)
        if 'p'+str(label) not in handles_list:
            handles_list.append('p'+str(label))
            labels_list.append(label) 
    # import ipdb; ipdb.set_trace()
    plt.legend([globals()[i] for i in handles_list], labels_list)
    # plt.legend([globals()[i] for i in handles_list])
    # plt.colorbar()
    plt.savefig('./result/ori_i3dData_tsne.png', dpi=120)
    plt.show()
                

def check_outliers(data_file):
    data_feat = h5py.File(os.path.join(BF_CONFIG["data_dir"], BF_CONFIG["feat_hdf5_name"]), 'r')
    data_nota = io.loads_json(os.path.join(BF_CONFIG["data_dir"], 'notation.json'))
    data_dir = [dir.rstrip() for dir in open(data_file).readlines()]
    for dir in data_dir:
        print(dir)
        # print training data
        label = data_feat[dir]['label'][:]
        print(label.shape)
        p = label[0]
        num = 0
        string = str()
        for l in label:
            if l == p:
                string += BF_ACTION_CLASS[l] + ' '
                num +=1
                p = l
            else:
                string += str(num)
                print(string)
                string = (BF_ACTION_CLASS[l] + ' ')
                num = 1
                p = l
        string += str(num)
        print(string)
        # print ground truth label
        nota = data_nota[dir]
        print(sorted(nota.items(), key=lambda a: int(a[0].split('-')[0])))
        print('-'*60 + '\n')
        
def img_enhance(data_dir, out_dir):
    # saturation
    random_factor = np.random.randint(0, 31) / 10.  
    color_image = ImageEnhance.Color(image).enhance(random_factor) 
    # brightness
    random_factor = np.random.randint(10, 21) / 10.  
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
    # contrast
    random_factor = np.random.randint(10, 21) / 10.  
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor) 
    # sharpness
    random_factor = np.random.randint(0, 31) / 10.  
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 

    res_img = sharp_image
    return res_img  

def tsne(): 
    from sklearn.datasets import load_digits   
    digits = load_digits()
    x, y = digits.data, digits.target
    # import ipdb; ipdb.set_trace()
    x_tsne = TSNE(n_components=2, init='pca', random_state=33).fit_transform(x)

    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) - (x_max - x_min)
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, label='t_SNE')
    plt.legend()
    plt.subplot(122)
    plt.scatter(x_norm[:, 0], x_norm[:, 1], c=y, label='t_SNE_norm')
    plt.legend()
    plt.savefig('./result/digits_tsne.png', dpi=120)
    plt.show()


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, '..')
    add_path(data_dir)
    from dataset.config import BF_CONFIG, BF_ACTION_CLASS
    import utils.io as io

    # vis_i3ddata_distribution()
    vis_i3ddata_tsne()
    # check_outliers('result/outlier_data.txt')

    # data1 = np.random.normal(0, 0.3, [1, 500])
    # data2 = np.random.normal(0, 0.1, [1, 500])
    # vis_distribution([data1, data2], legend=['0.3', '0.1'])
    # tsne()