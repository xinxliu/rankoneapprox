import torch
from torch import nn
from torch.autograd import Variable
import math

eps = 1e-8

class mf_sqrtFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,X):
        ctx.save_for_backward(X)
        return torch.sqrt(X)

    @staticmethod
    def backward(ctx,grad_output):
        X = ctx.saved_variables[0]
       # print(X.size())
      #  print(grad_output.size())
        grad_X = 0.5/torch.sqrt(X + eps)
        grad_X = grad_output.mul(grad_X)
        return grad_X, None

mf_sqrt = mf_sqrtFunction.apply

class mf_sign_sqrtFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,X):
        ctx.save_for_backward(X)
        output = torch.mul(torch.sign(X),torch.sqrt(torch.abs(X)))
        return output

    @staticmethod
    def backward(ctx,grad_output):
        X = ctx.saved_variables[0]
        grad_X = grad_output.mul(0.5/torch.sqrt(torch.abs(X)+eps))
        # uncomment if use grad_0 = 0
        # grad_X[X == 0] = 0
        return grad_X,None

mf_sign_sqrt = mf_sign_sqrtFunction.apply

class mf(torch.nn.Module):
    def __init__(self,k,d):
        super(mf,self).__init__()
        self._outdim = kl
        self._indim = d
        self.U = nn.Parameter(torch.Tensor(k,d))
        self.bias = nn.Parameter(torch.Tensor(k))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.U.size(1)) #512
        self.U.data.uniform_(-stdv,stdv)
        self.bias.data.uniform_(-stdv,stdv)
    def forward(self,x):
        x = x.view(x.size(0), self._indim, -1) #32*512*49
        z = torch.matmul(self.U, x) #32*200*49
        s = torch.squeeze(torch.sum(z * z, 2)) + self.bias
        return s

    def __repr__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'

    def __str__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'

class mfTwoPara(nn.Module):
    def __init__(self,k,d):
        super(mfTwoPara,self).__init__()
        self._outdim = k
        self._indim = d
        self.U = nn.Parameter(torch.Tensor(k,d))
        self.bias = nn.Parameter(torch.Tensor(k))
        self.V = nn.Parameter(torch.Tensor(k,d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.U.size(1))
        self.U.data.normal_(0,stdv)
        self.bias.data.normal_(0,stdv)
        self.V.data.normal_(0,stdv)

    def forward(self,x):
        x = x.view(x.size(0), self._indim, -1)
        z_1 = torch.matmul(self.U, x)
        z_2 = torch.matmul(self.V, x)
        s = torch.squeeze(torch.sum(z_1 * z_2, 2)) + self.bias
        return s

    def __repr__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'

    def __str__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'

class mfTwoStream(nn.Module):
    def __init__(self,k,d):
        super(mf,self).__init__()
        self.U = nn.Parameter(torch.Tensor(k,d))
        self.bias = nn.Parameter(torch.Tensor(k))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(math.sqrt(self.U.size(0)))
        self.U.data.uniform_(-stdv,stdv)
        #self.bias.data.uniform_(-stdv,stdv)
        self.bias = 0
    def forward(self,x_1,x_2):
        x_1 = x_1.view(x_1.size(0), 512, -1)
        x_2 = x_2.view(x_2.size(0), 512, -1)

        z_1 = torch.matmul(self.U, x_1)
        z_2 = torch.matmul(self.U, x_2)
        s = torch.squeeze(torch.sum(z_1 * z_2, 2)) + self.bias
        return s

    def __repr__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'

    def __str__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(0))+')'

class sign_sqrt_norm(torch.nn.Module):
    def __init__(self):
        super(sign_sqrt_norm,self).__init__()

    def forward(self, x):
        return mf_sign_sqrt(x)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__


class mfbi(torch.nn.Module):
    def __init__(self,k,d):
        super(mfbi,self).__init__()
        self._outdim = k #200
        self._indim = d  #512
        self.U = nn.Parameter(torch.Tensor(k,d))
        self.bias = nn.Parameter(torch.Tensor(k))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.U.size(1))
        self.U.data.normal_(0, stdv)
        self.bias.data.normal_(0, stdv)

    def forward(self, x):
        x = x.view(x.size(0), self._indim, -1)
        z = torch.matmul(self.U, x)
        y = torch.matmul(self.U * self.U, x*x)
        s = 1/2*torch.sum(z*z + y, 2) + self.bias
        return s

    def __repr__(self):
        return self.__class__.__name__+ '( UV: ' +str(self.U.size(0))+','+str(self.U.size(1))+')'

    def __str__(self):
        return self.__class__.__name__+ '(' +str(self.U.size(0))+','+str(self.U.size(1))+')'



class mfThreePara(nn.Module):
    def __init__(self,k,d):
        super(mfThreePara,self).__init__()
        self._outdim = k
        self._indim = d
        self.U = nn.Parameter(torch.Tensor(k,d))
        self.bias = nn.Parameter(torch.Tensor(k))
        self.C = nn.Parameter(torch.Tensor(k))
        self.V = nn.Parameter(torch.Tensor(k,d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.U)
        nn.init.constant(self.bias,0)
        nn.init.xavier_uniform(self.V)
        nn.init.constant(self.C,0)

    def forward(self,x):
        x = x.view(x.size(0), self._indim, -1)
        z_1 = torch.matmul(self.U, x)
        z_2 = torch.matmul(self.V, x)
        xx = torch.sum(x, 1) # 32*512*49 -> 32*49
        xx = xx*xx # 32*49 eq to torch.mul
        xx = torch.sum(xx, 1) # 32*49 -> 32*1
        c = self.C.expand(x.size(0),200)
        xx = xx.view(x.size(0),1).expand(x.size(0),200)
        cxx = c*xx # 32*200
        s = torch.squeeze(torch.sum(z_1 * z_2, 2)) + self.bias + cxx
        return s

    def __repr__(self):
        return self.__class__.__name__+ ' UVC: (' +str(self.U.size(0))+','+str(self.U.size(1))+')'

    def __str__(self):
        return self.__class__.__name__+ ' UVC: (' +str(self.U.size(0))+','+str(self.U.size(0))+')'