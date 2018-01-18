import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import mflayer
import torchvision.transforms as transforms


# vgg_m = torchvision.models.vgg19(pretrained=True)
vgg_d = torchvision.models.vgg16_bn(pretrained=True)
vgg_d_wobn = torchvision.models.vgg16(pretrained=True)

class Net(nn.Module):

    def __init__(self, net_type='vgg16', num_classes=200, TwoPara = True, classifier = False, sqrt_norm = True):
        super(Net, self).__init__()
        self._net_type = net_type
        self._num_classes = num_classes
        self._TwoPara = TwoPara
        self._classfier = classifier
        self._sqrt_norm = sqrt_norm

        self._indim, self._outdim = self._get_dim()
        # make module
        # self.features = self._make_features()
        self.features = vgg_d.features
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.mfTwoPara = mflayer.mfTwoPara(200, self._indim)  # 200 * 512

    def forward(self,X):
        x = self.features(X)  # bs*512*7*7
        x = self.ss_norm(x)        # bs*512*7*7
        s = self.mfTwoPara(x)
        s = F.normalize(s.view(x.size(0), -1))
        return s

    def _make_features(self):
        if self._net_type == 'vgg16':
            features = vgg_d_wobn.features
            features = nn.Sequential(*list(features)[:-1])
            return features
        elif self._net_type == 'resnet50':
            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet = list(resnet50.children())[:-2]
            res_features = torch.nn.Sequential(*resnet)
            return res_features
        print("net type error")

    def _get_dim(self):
        if self._net_type == "vgg16":
            return 512, 4096
        elif self._net_type == "resnet50":
            return 2048, 2048


    def _make_classifier(self):
        if self._net_type == "vgg16":
            return nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,self._num_classes)
            )
        elif self._net_type == "resnet50":
            return nn.Sequential(
                nn.Linear(2048, self._num_classes)
            )



class Baseline(nn.Module):
    def __init__(self):
        super(Baseline,self).__init__()
        self.features = vgg_d_wobn.features
        # self.classifier = nn.Linear(512, 200)
        self.classifier = self.get_classfier()

    def get_classfier(self):
        classifier_ = vgg_d_wobn.classifier
        classifier_._modules['6'] = torch.nn.Linear(4096, 200)
        return classifier_

    def forward(self, X):
        x = self.features(X)
        x = x.view(x.size(0), -1)
        s = self.classifier(x)
        return s

class regu_Net(nn.Module):

    def __init__(self, net_type='vgg16', num_classes=200, TwoPara = True, classifier = False, sqrt_norm = True, regu_lambda=1):
        super(regu_Net, self).__init__()
        self._net_type = net_type
        self._num_classes = num_classes
        self._TwoPara = TwoPara
        self._classfier = classifier
        self._sqrt_norm = sqrt_norm

        self.regu_lambda = regu_lambda

        self._indim, self._outdim = self._get_dim()
        # make module
        self.features = self._make_features()
        # self.features = vgg_d_wobn.features
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.mfTwoPara = mflayer.mfTwoPara(200, 512)  # 200 * 512

    def forward(self,X):
        x = self.features(X)  # bs*512*7*7
        x = self.ss_norm(x)        # bs*512*7*7
        s = self.mfTwoPara(x)
        s = F.normalize(s.view(x.size(0), -1))

        # calculate regu..
        top = torch.matmul(self.mfTwoPara.U, self.mfTwoPara.V.transpose(0, 1))  # 200*200
        top = torch.diag(top)  # 200
        top = top*top

        b1 = torch.sum(self.mfTwoPara.U * self.mfTwoPara.U, 1)  # 200
        b2 = torch.sum(self.mfTwoPara.V * self.mfTwoPara.V, 1)  # 200
        bottom = b1 * b2

        similar = top / bottom
        reg = self.regu_lambda * torch.sum(similar) / 200
        return s, reg

    def _make_features(self):
        if self._net_type == 'vgg16':
            features = vgg_d_wobn.features
            features = nn.Sequential(*list(features)[:-1])
            return features

    def _get_dim(self):
        if self._net_type == "vgg16":
            return 512, 200

class tanh_Net(nn.Module):

    def __init__(self, net_type='vgg16', num_classes=200, TwoPara = True, classifier = False, sqrt_norm = True, regu_lambda=1):
        super(tanh_Net, self).__init__()
        self._net_type = net_type
        self._num_classes = num_classes
        self._TwoPara = TwoPara
        self._classfier = classifier
        self._sqrt_norm = sqrt_norm

        self.regu_lambda = regu_lambda
        self.alpha = 1

        self._indim, self._outdim = self._get_dim()
        # make module
        self.features = self._make_features()
        # self.features = vgg_d_wobn.features
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.mfTwoPara = mflayer.mfTwoPara(200, 512)  # 200 * 512

    def adjust_alpha(self, new):
        self.alpha = self.alpha + new

    def forward(self,X):
        x = self.features(X)  # bs*512*7*7
        x = self.ss_norm(x)        # bs*512*7*7
        s = self.mfTwoPara(x)
        s = F.normalize(s.view(x.size(0), -1))

        # calculate regu..
        top = torch.matmul(self.mfTwoPara.U, self.mfTwoPara.V.transpose(0, 1))  # 200*200
        top = torch.diag(top)  # 200
        top = top*top

        b1 = torch.sum(self.mfTwoPara.U * self.mfTwoPara.U, 1)  # 200
        b2 = torch.sum(self.mfTwoPara.V * self.mfTwoPara.V, 1)  # 200
        bottom = b1 * b2

        similar = top / bottom
        reg = self.regu_lambda * torch.sum(similar) / 200
        return s, reg

    def _make_features(self):
        if self._net_type == 'vgg16':
            features = vgg_d_wobn.features
            features = nn.Sequential(*list(features)[:-1])
            return features

    def _get_dim(self):
        if self._net_type == "vgg16":
            return 512, 200


class BCNN(nn.Module):
    def __init__(self):
        super(BCNN,self).__init__()
        self.features = self._get_features()
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.fc = nn.Linear(512*512, 200)

    def _get_features(self):
        features = vgg_d_wobn.features
        features = nn.Sequential(*list(features)[:-1])
        return features

    def forward(self, X):
        x = self.features(X)                    # bs*512*14*14
        x = x.view(x.size(0), 512, -1)
        x_t = torch.transpose(x, 1, 2)
        phi = torch.matmul(x, x_t)              # bs*512*512
        s = self.ss_norm(phi)                   # bs*512*512
        s = s.view(x.size(0), -1)
        s = F.normalize(s)
        s = self.fc(s)
        return s


# ----------------------abandoned solutions-------------------------
class MfbiNet(nn.Module):

    def __init__(self):
        super(MfbiNet, self).__init__()
        self.features = vgg_d.features
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.mfbi_1 = mflayer.mfbi(200, 512)
        self.mfbi_2 = mflayer.mfbi(200, 512)

    def forward(self,X):
        x = self.features(X)
        x = F.tanh(x)
        # x = self.ss_norm(x)
        # Mfbi
        s_1 = self.mfbi_1(x)
        #s_1 = F.normalize(s_1)
        y_1 = F.softmax(s_1)
        s_2 = self.mfbi_2(x)
        #s_2 = F.normalize(s_2)
        y_2 = F.softmax(s_2)
        y = 1/2*(y_1 + y_2)
        y = torch.log(y)
        return y


class ThreePara(nn.Module):

    def __init__(self, net_type='vgg16', num_classes=200, TwoPara = True, classifier = False, sqrt_norm = True):
        super(ThreePara, self).__init__()
        self._net_type = net_type
        self._num_classes = num_classes
        self._TwoPara = TwoPara
        self._classfier = classifier
        self._sqrt_norm = sqrt_norm

        # make module
        self.features = self._make_features()
        self.ss_norm = mflayer.sign_sqrt_norm()
        self.mfThreePara = mflayer.mfThreePara(200, 512)


    def forward(self,X):
        '''
        :param X: (C*H*W)images
        :return: res : score vector
        '''
        x = self.features(X)  # bs*512*7*7

        x = self.ss_norm(x)        # bs*512*7*7

        s = self.mfThreePara(x)

        # L2 Norm after mf
        s = F.normalize(s)
        return s

    def _make_features(self):
        if self._net_type == 'vgg16':
            vgg_d = torchvision.models.vgg16_bn(pretrained=True)
            return vgg_d.features
        elif self._net_type == 'resnet50':
            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet = list(resnet50.children())[:-2]
            res_features = torch.nn.Sequential(*resnet)
            return res_features
        elif self._net_type == "baseline":
            vgg_d = torchvision.models.vgg16_bn(pretrained=True)
            return vgg_d.features
        print("net type error")



if __name__ == "__main__":
    normalize = transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
    resize = transforms.Resize(256)
    randomcrop = transforms.RandomCrop(224)
    transform = transforms.Compose([resize,randomcrop,transforms.ToTensor,normalize])


