import mfcnn
import torchvision.transforms as transforms
import load_cub
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import test_mfcnn
import argparse


class Args(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self.net_type = cfg['net_type']
        self.lr = cfg['lr']
        self.globa_lr = cfg['glr']
        self.batch_size = cfg['bs']
        self.numEpoch = cfg['numEpoch']
        self.numClassifyEpoch = cfg['numClassifyEpoch']
        self.twoPara = False
        self.ss_norm = False
        if 'twoPara' in cfg.keys():
            self.twoPara = cfg['twoPara']
        if 'ss_norm' in cfg.keys():
            self.ss_norm = cfg['ss_norm']
        if 'cfg' in cfg.keys():
            self.cfg = cfg['cfg']
        if 'ts' in cfg.keys():
            self.ts = cfg['ts']

    def print(self):
        print(self._cfg)


def adjust_learning_rate(optimizer_, epoch_, args_):
    lr = args_.lr*(0.1**(epoch_//20))
    for param_group in optimizer_.param_groups:
        param_group['lr'] = lr
    print('epoch %d start, learning rate is %f' % (epoch_, lr))


def train(args):
    # 0
    normalize_0 = transforms.Normalize(mean=[0.486, 0.500, 0.433], std=[0.232, 0.228, 0.266])
    resize_0 = transforms.Scale(256)
    crop_0 = transforms.RandomCrop(224)
    crop_0_test = transforms.CenterCrop(224)
    flip_0 = transforms.RandomHorizontalFlip()
    transform_0 = transforms.Compose([resize_0, flip_0, crop_0, transforms.ToTensor(), normalize_0])
    transform_test_0 = transforms.Compose([resize_0, crop_0_test, transforms.ToTensor(), normalize_0])
    # 1
    normalize_1 = transforms.Normalize(mean=[0.486, 0.500, 0.433], std=[0.232, 0.228, 0.266])
    resize_1 = transforms.Scale(224)
    crop_1 = transforms.CenterCrop(224)
    flip_1 = transforms.RandomHorizontalFlip()
    transform_1 = transforms.Compose([resize_1, flip_1, crop_1, transforms.ToTensor(), normalize_1])
    transform_test_1 = transforms.Compose([resize_1, crop_1, transforms.ToTensor(), normalize_1])
    # 2
    normalize_2 = transforms.Normalize(mean=[0.486, 0.500, 0.433], std=[0.232, 0.228, 0.266])
    resize_2 = transforms.Scale(448)
    flip = transforms.RandomHorizontalFlip()
    crop_2 = transforms.RandomCrop(448)
    crop_2_test = transforms.CenterCrop(448)
    transform_2 = transforms.Compose([resize_2, flip, crop_2, transforms.ToTensor(), normalize_2])
    transform_test_2 = transforms.Compose([resize_2, crop_2_test, transforms.ToTensor(), normalize_2])
    if hasattr(args,'ts'):
        if args.ts == 1:
            transform = transform_1
            transform_test = transform_test_1
        elif args.ts == 2:
            transform = transform_2
            transform_test = transform_test_2
    else:
        transform = transform_0
        transform_test = transform_test_0

    trainset = load_cub.CUB200(transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = load_cub.CUB200(train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    tanh = False
    if hasattr(args,'cfg'):
        if args.cfg == 'Mfbi':
            net = mfcnn.MfbiNet()
            criterion = nn.NLLLoss().cuda()
        if args.cfg == 'baseline':
            net = mfcnn.Baseline()
            criterion = nn.CrossEntropyLoss().cuda()
        if args.cfg == 'tp':
            net = mfcnn.ThreePara()
            criterion = nn.CrossEntropyLoss().cuda()
        if args.cfg == 'bcnn':
            net = mfcnn.BCNN()
            criterion = nn.CrossEntropyLoss().cuda()
        if args.cfg == 'reg_mfcnn':
            net = mfcnn.regu_Net()
            criterion = nn.CrossEntropyLoss().cuda()
        if args.cfg == 'tanh_mfcnn':
            net = mfcnn.tanh_Net()
            criterion = nn.CrossEntropyLoss().cuda()
            tanh = True
    else:
        net = mfcnn.Net(net_type=args.net_type, TwoPara=args.twoPara)
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    args.print()

    net.features = torch.nn.DataParallel(net.features)
    print("Paralleling model's feature  .....")

    print(net)
    net = net.cuda()

    log_loss = []
    log_prec_1 = []

    # fix params in vgg-features
    for params in net.features.parameters():
        params.requires_grad = False

    for epoch in range(args.numEpoch):
        # prepare stastic tool
        losses = test_mfcnn.AverageMeter()
        top1 = test_mfcnn.AverageMeter()
        top5 = test_mfcnn.AverageMeter()
        # switch to train mode
        net.train()
        if epoch < args.numClassifyEpoch:
            adjust_learning_rate(optimizer, epoch, args)

        if tanh:
            if net.alpha <= 4:
                if (epoch + 1) % 10 == 0:
                    net.adjust_alpha(1)

        if epoch == args.numClassifyEpoch:
            for params in net.features.parameters():
                params.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.globa_lr

        for i, data in enumerate(trainloader):
            # prepare input and label
            inputs, labels = data
            labels = labels.cuda()
            inputs_var = Variable(inputs.cuda())
            target_var = Variable(labels)

            # compute output and loss


            if args.cfg == 'reg_mfcnn':
                outputs, reg = net(inputs_var)
                loss = criterion(outputs,target_var)
                loss += reg
            else:
                outputs = net(inputs_var)
                loss = criterion(outputs, target_var)

            # measure accuracy and record loss
            prec1, prec5 = test_mfcnn.accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 80 == 79:
                print('Epoch:[{0}][{1}/{2}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(trainloader), loss=losses, top1=top1, top5=top5))
        test_mfcnn.test_mfcnn(net, testloader, criterion)

        log_loss.append((epoch, losses))
        log_prec_1.append((epoch, top1))
    return log_loss, log_prec_1, net.state_dict()


if __name__ == "__main__":

    cfg0 = {'net_type': 'baseline', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.001,
            'glr': 0.00001}

    cfg1 = {'net_type':'vgg16', 'bs':32,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001}

    cfg2 = {'net_type':'vgg16', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001}

    cfg3 = {'net_type':'vgg16', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
            'extra':'two_FCs'}

    cfg5 = {'net_type':'vgg16', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
            'extra':'one_FC,sqrt_norm'}

    cfg6 = {'net_type':'resnet50', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
            'extra':'sqrt_norm,one_FC'}

    cfg7 = {'net_type':'vgg16', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
            'extra':'one_FC'}

    cfg9 = {'net_type':'resnet50', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
            'extra': 'one_FC'}

    cfg10 = {'net_type':'resnet50', 'bs':64,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001}

    cfg11 = {'net_type':'vgg16', 'bs':32,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
             'extra':'two_FCs'}

    cfg12 = {'net_type':'vgg16', 'bs':32,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
             'twoPara':True}

    cfg13 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'extra':'L2norm Embedding'}

    cfg14 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.00001,
             'twoPara':True,'ss_norm':True, 'extra':'L2norm'}     # 73.2%

    cfg15 = {'net_type':'vgg16', 'bs':32,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
             'cfg':'Mfbi','extra':'L2norm'}

    cfg16 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True,'ts':2, 'extra':'448, train_centercrop, resize_448'} # 77%

    cfg17 = {'net_type':'vgg16', 'bs':32,'numEpoch':100,'numClassifyEpoch':40,'lr':0.1,'glr':0.0001,
             'cfg':'Mfbi','ts':1,'extra':'tanh, no L2 norm'}

    cfg18 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'ts':1,'extra':'L2norm(chan-wise) following ss_norm'} # 70.1%

    cfg19 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True,'ts':1, 'extra':'ss_norm, grad_X[X==0]=0'}

    cfg20 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'ts':1,'extra':'L2norm(ele-wise) following ss_norm'}

    cfg21 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.01,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'ts':1,'extra':'L2norm(ele-wise) following ss_norm + bn'}

    cfg22 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'extra':'L2norm(ele-wise) lr-init = 0.1 '}

    cfg23 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'extra':'dim-reduce 256'}

    cfg24 = {'net_type':'vgg16','cfg':'baseline','bs':32,'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.001,
            'glr': 0.00001}

    cfg25 = {'net_type': 'vgg16', 'cfg': 'baseline', 'bs': 32, 'numEpoch': 40, 'numClassifyEpoch': 30, 'lr': 0.001,
             'glr': 0.00001}

    cfg26 = {'net_type': 'vgg16', 'cfg': 'baseline', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.001,
             'glr': 0.00001}

    cfg27 = {'net_type':'vgg16', 'cfg':'tp','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.0001,
             'twoPara':True,'ss_norm':True, 'extra':'L2norm'}

    cfg28 = {'net_type': 'vgg16', 'cfg': 'tp', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.1,
             'glr': 0.0001, 'extra': 'L2norm'}

    cfg29 = {'net_type':'vgg16','bs':32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.00001,
             'twoPara':True,'ss_norm':True, 'extra':'L2norm'}

    cfg30 = {'net_type': 'vgg16', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.1, 'glr': 0.00001,
             'twoPara': True, 'ss_norm': True, 'extra': 'L2norm'}

    cfg31 = {'net_type': 'vgg16', 'cfg': 'bcnn', 'bs': 32, 'numEpoch': 200, 'numClassifyEpoch': 60, 'lr': 0.9,
             'glr': 0.001}  # 80.1%

    cfg32 = {'net_type': 'vgg16', 'cfg': 'bcnn', 'bs': 32, 'numEpoch': 150, 'numClassifyEpoch': 45, 'lr': 0.9,
             'glr': 0.001, 'ts': 2}  # 84%

    cfg33 = {'net_type':'vgg16','bs': 32,'numEpoch':100,'numClassifyEpoch':30,'lr':0.1,'glr':0.00001,
             'twoPara':True,'ss_norm': True, 'extra':'L2norm'}

    cfg34 = {'net_type': 'vgg16', 'cfg': 'reg_mfcnn', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.1, 'glr': 0.00001,
             'twoPara': True, 'ss_norm': True, 'extra': 'L2norm'}

    cfg35 = {'net_type': 'vgg16', 'cfg': 'reg_mfcnn', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.1,
             'glr': 0.00001, 'twoPara': True, 'ss_norm': True, 'extra': 'L2norm, 2'}

    cfg36 = {'net_type': 'vgg16', 'cfg': 'reg_mfcnn', 'bs': 32, 'numEpoch': 100, 'numClassifyEpoch': 30, 'lr': 0.1,
             'glr': 0.00001, 'twoPara': True, 'ss_norm': True, 'extra': 'L2norm, 2', 'ts': 2}

    _args_ = "cfg36"

    _args = Args(eval(_args_))
    # modify when use a new net
    # vgg16, resnet50, baseline

    loss, top1, dict = train(_args)
    # modify when use a new cfg
    torch.save(loss, "./logs/" + _args_ + ".loss")
    torch.save(top1, "./logs/" + _args_ + ".top1")
    # torch.save(net, "./models/" + _args_ + ".pth")
    torch.save(dict, "./models/" + _args_ + "_dict.pth")



