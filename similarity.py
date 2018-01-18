import torch
import matplotlib.pyplot as plt
import numpy as np

UV = torch.load('./result/cos2_UV.pth')
U = UV['U']
V = UV['V']

top = np.dot(U, V.transpose())  # 200*200
top = np.diag(top)              # 200

b1 = np.sum(U*U, 1)             # 200
b2 = np.sum(V*V, 1)             # 200
bottom = np.sqrt(b1*b2)

similar = top/bottom            # 200
cos = np.arccos(similar)*180/np.pi

plt.figure()
plt.subplot(211)
plt.hist(similar)
# plt.hist(similar,cumulative=True,normed=True)
plt.title('similarity of U & V')
plt.subplot((212))
plt.hist(cos)

plt.savefig('./result/similarity_reg_cos2.png')
plt.show()
