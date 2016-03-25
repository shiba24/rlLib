import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class FcNN3(chainer.Chain):
    """ Fully-connected neural network of 3 layers. """
    def __init__(self, n_in, n_units, n_out):
        super(FcNN3, self).__init__(
            l1=L.Linear(n_in, n_units, wscale=np.sqrt(2.0)),
            l2=L.Linear(n_units, n_units, wscale=np.sqrt(2.0)),
            l3=L.Linear(n_units, n_out, wscale=np.sqrt(2.0)),
       )
        self.train = True
        self.history = np.array([np.zeros(n_out)])

    def __call__(self, x, t, index):
        h = self.predict(x)
        self.history = np.append(self.history, np.array([np.mean(h.data, axis=0)]), axis=0)

        h = F.select_item(h, index)             # choose the action[index] in each column
        error_abs = abs(h - t)
        error = F.concat((F.expand_dims(error_abs ** 2, 1), F.expand_dims(error_abs, 1)), axis=1)
        # 1 < error_abs <=> error ** 2 > error,  error < 1 <=> error ** 2 < error
        self.loss = F.sum(F.min(error, axis=1)) / np.float32(len(error_abs))
        return self.loss

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


"""

        # self.loss = F.mean_squared_error(h,t)     # before

#####  MEMO
###########
h_act = np.array([h.data[x, index[x]] for x in range(len(index))]).astype(np.float32)

self.accuracy = F.accuracy(h, t)
        # self.loss = F.clipped_relu(F.mean_squared_error(h, t), z=1.0)
        # h = F.dropout(F.clipped_relu(self.l1(x), z=20.0), train=self.train)
        # h = F.dropout(F.clipped_relu(self.l2(h), z=20.0), train=self.train)

print(h_act)
print(h.data)
self.loss = F.mean_squared_error(chainer.Variable(h_act), t)

        # print((h ** 2).data, (h - t).data)
        # h = F.reshape(h, (len(h.data), 1))
        # t = F.reshape(t, (len(t.data), 1))
        # print(abs(h - t), abs(h - t).data)
        ht_abs = abs(h - t)
        ht = F.concat((F.expand_dims(ht_abs ** 2, 1), F.expand_dims(ht_abs, 1)), axis=1)
        # print(F.min(ht, axis=1).data)
        self.loss = F.sum(F.min(ht, axis=1)) / np.float32(len(ht_abs))
        # print(F.min((ht_abs ** 2, ht_abs), axis=1))
        # print((ht_abs * ht_abs * (ht_abs.data <= 1)).data)
        # k = ht_abs * ht_abs * (ht_abs.data <= 1) + ht_abs * (ht_abs.data > 1)
        # print(k.data, k.data.shape)
        # print(F.expand_dims(k, 1).data)
        # print(F.sum(k, axis=0).data)
# td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        # print(F.reshape(F.max(F.concat((t - h, h - t), axis=1), axis=1), (len(t.data), 1)).data)
        # ht_abs = F.reshape(F.max(F.concat((t - h, h - t), axis=1), axis=1), (len(t.data), 1))
        # self.loss = F.min((ht_abs ** 2, ht_abs), axis=1)
        # self.loss = F.sum(ht_abs * ht_abs * (ht_abs.data <= 1) + ht_abs * (ht_abs.data > 1)) / chainer.Variable(np.array([len(ht_abs)]).astype(np.float32))
        # self.loss = F.mean_squared_error(h,t)

"""
