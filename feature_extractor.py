import torch
import torch.nn as nn
import mfcnn
import torchvision.transforms as transforms
import load_cub
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy

if __name__ == "__main__":
    net = mfcnn.Net()
    net.features = nn.DataParallel(net.features)
    d = torch.load('./models/cfg29_dict.pth')
    net.load_state_dict(d)
    net = net.cuda()
    net.eval()

    bcnn_net = mfcnn.BCNN()
    bcnn_net.features = nn.DataParallel(bcnn_net.features)
    bcnn_d = torch.load('./models/cfg32_dict.pth')
    bcnn_net.load_state_dict(bcnn_d)
    net = net.cuda()
    net.eval()

    fea = net.features
    bcnn_fea = net.features


    use_gpu = True
    name_model = './models/cfg29_dict.pth'
    bcnn_model = './models/cfg32_dict.pth'
    # =======================
    normalize_0 = transforms.Normalize(mean=[0.486, 0.500, 0.433], std=[0.232, 0.228, 0.266])
    resize_0 = transforms.Scale(256)
    crop_0_test = transforms.CenterCrop(224)
    flip_0 = transforms.RandomHorizontalFlip()
    transform = transforms.Compose([resize_0, crop_0_test, transforms.ToTensor(), normalize_0])

    # ==========================================
    normalize_2 = transforms.Normalize(mean=[0.486, 0.500, 0.433], std=[0.232, 0.228, 0.266])
    resize_2 = transforms.Scale(448)
    flip = transforms.RandomHorizontalFlip()
    crop_2 = transforms.RandomCrop(448)
    crop_2_test = transforms.CenterCrop(448)
    transform_2 = transforms.Compose([resize_2, crop_2_test, transforms.ToTensor(), normalize_2])
    # ===========================================
    testset = load_cub.CUB200(train=False, transform=transform)
    testset_bcnn = load_cub.CUB200(train=False, transform=transform_2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
    testloader_bcnn = torch.utils.data.DataLoader(testset_bcnn, batch_size=32, shuffle=True, num_workers=4)

    plt.close('all')
    r_ = []
    for i, data in enumerate(testloader):
        input, label = data
        input = Variable(input.cuda())
        label = label.cuda()
        output = fea(input)  # bs*200
        r = numpy.random.randint(0, input.size(0))
        r_.append(r)
        x = output.data.cpu().numpy()[r]
        x = x.reshape(x.size)
        plt.figure()
        plt.subplot(211)
        plt.hist(x, 200, normed=True)
        plt.subplot(212)
        plt.hist(x[x < 0.005], 200, normed=True)
        plt.title('mfcnn')
        plt.savefig(str(i) + '_' + str(r) + '_' + 'mfcnn' + '.png')

    for i, data in enumerate(testloader):
        input, label = data
        input = Variable(input.cuda())
        label = label.cuda()
        output = bcnn_fea(input)  # bs*200
        r = r_[i]
        x = output.data.cpu().numpy()[r]
        x = x.reshape(x.size)
        plt.figure()
        plt.subplot(211)
        plt.hist(x, 200, normed=True)
        plt.subplot(212)
        plt.hist(x[x < 0.005], 200, normed=True)
        plt.title('bcnn')
        plt.savefig(str(i) + '_' + str(r) + '_' + 'bcnn' + '.png')

