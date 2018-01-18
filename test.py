from torch.autograd import Variable
import torch.autograd
import torch
import numpy as np

x_1 = Variable(torch.randn(512,49),requires_grad=True)
x_2 = Variable(torch.randn(512,49),requires_grad=True)
u = Variable(torch.rand(200,512),requires_grad=True)
bias = Variable(torch.rand(200,1),requires_grad=True)
z_1 = torch.mm(u,x_1)
z_2 = torch.mm(u,x_2)
z = z_1*z_2
s_temp = torch.sum(z_1*z_2,1)
s = s_temp.view(200,1)
print(s.size())
s = s + bias
p = torch.sigmoid(s)
target = Variable(torch.randn(200,1))

loss = torch.nn.MSELoss()
print(z.size())
print(bias.size())
print(s.size())
print(p.size())
output = loss(p,target)
output.backward()
print(s_temp.grad_fn)
print(s.grad_fn)




