import cv2
import numpy as np
from os.path import splitext
from aoss_client.client import Client
import random
import os


def saveimg(imgname, data_root, local_root, client, label=False):
    img_url = data_root+imgname
    img_bytes = client.get(img_url)
    assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if label:
        img = img*255
    save_imgpath = os.path.join(local_root, imgname)
    save_imgdir = '/'.join(save_imgpath.split('/')[:-1])
    os.makedirs(save_imgdir, exist_ok=True)
    cv2.imwrite(save_imgpath, img)

def get_size(imgname, data_root, local_root, client, label=False):
    img_url = data_root+imgname
    img_bytes = client.get(img_url)
    assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if label:
        img = img*255
    return img.shape



if __name__=='__main__':
    conf_path = '/home/SENSETIME/wangyichen/Downloads/aoss-python-sdk-2.2.6/aoss.conf'
    client = Client(conf_path) # 若不指定 conf_path ，则从 '~/aoss.conf' 读取配置文件
    # img_url = 'sh36_ssd:s3://3darseg.segmentation/tt.jpg'
    # img_url = "sh36_ssd:s3://3darseg.root/segmentation/skin/lists/details/2025000408-skin-light_train.txt"
    # img_bytes = client.get(img_url)
    # assert(img_bytes is not None)
    # img_mem_view = memoryview(img_bytes)
    # img_array = np.frombuffer(img_mem_view, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # cv2.imwrite('/home/SENSETIME/wangyichen/Documents/codes/st_000617.jpg', img)
    sample_num = 15

    # url = "sh36_ssd:s3://3darseg.root/segmentation/skin/2017data/image/"
    # txt_url = 'sh36_ssd:s3://3darseg.root/segmentation/skin/lists/details/181020_train.txt'
    data_root = 'bj17:s3://3darseg.segmentation/sod'
    local_root = '/home/SENSETIME/wangyichen/Documents/codes/segdata_samples'
    txt_list = [
        'bj17:s3://3darseg.segmentation/sod/cat_dog_pseudo/pseudo_lists/trainval_cat_dog_2.txt']
    for txt_url in txt_list:
        
        data = client.get(txt_url)
        image_list = str(data).split('\\n')
        sample_list = random.sample(image_list, sample_num)
        print(f'{txt_url}, {len(image_list)}')
        for sample in image_list:
            imgname, labelname = sample.split(' ')
            img_size = get_size(imgname, data_root, local_root, client)
            print(img_size)
