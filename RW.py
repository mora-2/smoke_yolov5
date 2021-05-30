import imghdr
import os
import numpy as np

if __name__ == "__main__":
    # rootdir = './data/images'
    # list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    # for i in range(0, len(list)):
    #     path = os.path.join(rootdir, list[i])
    #     # if os.path.isfile(path):
    #     # # 你想对文件的操作
    #     print(path)
    #     print(imghdr.what(path))
    #     if imghdr.what(path) == 'gif':
    #         os.remove(path)

    # a = np.load('./data/labels.npy', allow_pickle=True)
    # print(a)

    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
        mixed_precision = False  # not installed
