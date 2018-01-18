import torch
import matplotlib.pyplot as plt

UV = torch.load('./UV.pth')
U = UV['U']
U = U.reshape(U.size)
V = UV['V']
V = V.reshape(V.size)

init_UV = torch.load('./init_UV.pth')
init_U = init_UV['U']
init_U = init_U.reshape(init_U.size)
init_V = init_UV['V']
init_V = init_V.reshape(init_V.size)

bcnn_w = torch.load('./bcnn_w.pth')

bins = 100
plt.figure()
plt.subplot(211)
plt.hist(U, bins, normed=True)
plt.subplot(212)
plt.hist(init_U, bins, normed=True)
plt.savefig('./u.png')
plt.figure()
plt.subplot(211)
plt.hist(V, bins, normed=True)
plt.subplot(212)
plt.hist(init_V, bins, normed=True)
plt.savefig('./v.png')
plt.figure()
plt.hist(bcnn_w, bins, normed=True)
plt.savefig('./bcnn_w.png')

plt.show()


