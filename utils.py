from torch.autograd import Variable
import torchvision.transforms as transforms

cub_mean = [0.486, 0.500, 0.433]
cub_std = [0.232, 0.228, 0.266]


class Transform(object):
    ''' create transforms for vision tasks
    Usage:
    t = Transform(mean_cub=(.,.))
    transform = t.getTransform(resize, croptype, cropsize, train)
    resize: int, Scale size
    croptype: string "train" or else
    cropsize: int crop_size
    train: bool, if True, there will be a random flip
    '''
    def __init__(self, mean_std=(cub_mean, cub_std)):
        self._flip = transforms.RandomHorizontalFlip()
        self._Scale = transforms.Scale(224)
        self._Crop = transforms.RandomCrop(224)
        self._CenterCrop = transforms.CenterCrop(224)
        self._tt = transforms.ToTensor()
        self._normalize = transforms.Normalize(mean=mean_std[0], std=mean_std[1])

        self.transform_224_train = transforms.Compose(
            [self._Scale, self._flip, self._Crop, self._tt, self._normalize]
        )
        self.transform_224_test = transforms.Compose(
            [self._Scale, self._Crop, self._tt, self._normalize]
        )

    def _fScale(self, s):
        return transforms.Scale(s)

    def _fCrop(self, s, random=True):
        if random:
            return transforms.RandomCrop(s)
        else:
            return transforms.CenterCrop(s)

    def getTransform(self, resize, croptype, cropsize, train):
        if croptype == 'random':
            if train:
                return transforms.Compose([
                    self._fScale(resize), self._flip, self._fCrop(cropsize), self._tt, self._normalize
                ])
            else:
                return transforms.Compose([
                    self._fScale(resize), self._fCrop(cropsize), self._tt, self._normalize
                ])
        else:
            if train:
                return transforms.Compose([
                self._fScale(resize), self._flip, self._fCrop(cropsize, random=False), self._tt, self._normalize
                ])
            else:
                return transforms.Compose([
                    self._fScale(resize), self._fCrop(cropsize, random=False), self._tt, self._normalize
                ])


def test_model(model, testloader, criterion, topk = (1,)):
    '''
    calculate acc and loss of a model then print them
    :param model: pytorch model
    :param testloader: testloader
    :param criterion: to calculate loss
    :param topk, topk acc
    :return acc(list, topk), average loss

    '''
    # necessary if there is bn
    model.eval()
    loss = 0
    acc = [0 for i in range(len(topk))]
    count = 0
    for imgs, labels in testloader:
        imgs = Variable(imgs.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        N = imgs.size(0)
        count += N
        output = model(imgs)
        lossinepoch = criterion(output, labels)
        loss += lossinepoch * N

        res = accuracy(output, labels, topk)
        res = [i*N for i in res]
        assert len(res) == len(acc)
        acc = [i+j for i, j in zip(acc, res)]

    loss = loss / count
    acc = [i / count for i in acc]
    return acc, loss


def accuracy(output,target,topk=(1,)):
    '''
    calculate average topk acc for classification tasks
    :param output: score vector
    :param target: labels
    :param topk: tuple ,its elements should be `int`
    :return: acc, its length is equal to tuple topk
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
