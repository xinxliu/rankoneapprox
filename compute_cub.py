from PIL import Image
import numpy as np
import load_cub
import math
import torch

if __name__ == '__main__':
    train_set = load_cub.CUB200()
    imgs = train_set.imgs
    imgs_path = []
    img_mats = []
    for path,idx in imgs:
        imgs_path.append(path)
        img = Image.open(path)
        img.convert('RGB')
        img_mats.append(np.array(img))

    count = 0
    sum_r = 0
    sum_g = 0
    sum_b = 0

    for i, mat in enumerate(img_mats):

        if len(mat.shape) == 3:
            count += mat.size/3
            sum_r += np.sum(mat[:,:,0])
            sum_g += np.sum(mat[:,:,1])
            sum_b += np.sum(mat[:,:,2])
        else:
            print("not rgb: %d" % i)
    mean = (sum_r/count,sum_g/count,sum_b/count)

    sum_sq_r = 0
    sum_sq_g = 0
    sum_sq_b = 0
    count = 0

    for i,mat in enumerate(img_mats):
        if len(mat.shape) == 3:
            count += mat.size/3
            mat = mat - np.array(mean).reshape(1,1,3)
            sum_sq_r += np.sum(mat[:,:,0]*mat[:,:,0])
            sum_sq_g += np.sum(mat[:,:,1]*mat[:,:,1])
            sum_sq_b += np.sum(mat[:,:,2]*mat[:,:,2])
        else:
            print("not rgb: %d" % i)
    std = (math.sqrt(sum_sq_r/count),math.sqrt(sum_sq_g/count),math.sqrt(sum_sq_b/count))
    mean_std = (mean,std)
    print(mean_std)
    torch.save(mean_std,'cub_mean_std.pth')


