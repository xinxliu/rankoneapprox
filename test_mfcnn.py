import mfcnn
import torch
from torch.autograd import Variable
import load_cub
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy
import matplotlib
import matplotlib.pyplot as plt



def test_mfcnn(model,testloader,criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i,(input,target) in enumerate(testloader):
        target = target.cuda()
        input_var = Variable(input.cuda(),volatile = True)
        target_var = Variable(target,volatile = True)

        output, _ = model(input_var)
        loss = criterion(output,target_var)

        prec1,prec5 = accuracy(output.data,target,topk=(1,5))
        losses.update(loss.data[0],input.size(0))
        top1.update(prec1[0],input.size(0))
        top5.update(prec5[0],input.size(0))

    print('Test: * Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(loss=losses,top1=top1,top5=top5))

def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count



if __name__ == "__main__":

    #=========cfg===========
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    use_gpu = True
    name_model = './models/cfg29_dict.pth'
    bcnn_model = './models/cfg32_dict.pth'
    #=======================
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

    correct = 0
    total = 0

    net = mfcnn.Net()
    net_bcnn = mfcnn.BCNN()
    d = torch.load(name_model)
    d_bcnn = torch.load(bcnn_model)
    new_d = OrderedDict()
    if use_gpu:
        net.features = torch.nn.DataParallel(net.features)
        net.load_state_dict(d)
        net = net.cuda()

        net_bcnn.features = torch.nn.DataParallel(net_bcnn.features)
        net_bcnn.load_state_dict(d_bcnn)
        net_bcnn = net_bcnn.cuda()
    else:
        for k,v in d.items():
            if 'module' in k:
                name = k[:9] + k[16:]
                new_d[name] = v
            else:
                new_d[k] = v
        net.load_state_dict(new_d)
        net = net.cpu()
    net.eval()

    plt.close('all')
    r_ = []
    for i, data in enumerate(testloader):
        input, label = data
        input = Variable(input.cuda())
        label = label.cuda()
        output = net(input)  # bs*200
        _, pred = output.data.max(1)  # bs*1
        pred = torch.squeeze(pred)
        r = numpy.random.randint(0, input.size(0))
        r_.append(r)
        s = output.data.cpu().numpy()[r]
        plt.figure()
        plt.subplot(211)
        plt.plot(s)
        res = ''
        if pred[r] == label[r]:
            res = 'right'
        else:
            res = 'wrong'
        plt.title(res)
        plt.subplot(212)
        plt.hist(s, 20, normed=True)
        plt.savefig(str(i) + '_' + str(r) + '_' + 'mfcnn_' + res + '.png')

    for i, data in enumerate(testloader_bcnn):
        input, label = data
        input = Variable(input.cuda())
        label = label.cuda()
        output = net_bcnn(input)  # bs*200
        _, pred = output.data.max(1)  # bs*1
        pred = torch.squeeze(pred)
        r = r_[i]
        s = output.data.cpu().numpy()[r]
        plt.figure()
        plt.subplot(211)
        plt.plot(s)
        res = ''
        if pred[r] == label[r]:
            res = 'right'
        else:
            res = 'wrong'
        plt.title(res)
        plt.subplot(212)
        plt.hist(s, 20, normed=True)
        plt.savefig(str(i) + '_' + str(r) + '_' + 'bcnn_' + res + '.png')
    plt.show()

