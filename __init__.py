import keras
import theano
import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Input, Lambda, merge, Layer

class A(Layer):
    
    # inputs are L and U and mu (in that order)
    def __init__(self, u_size, **kwargs):
        self.u_size = u_size
        self.lower_size = int(u_size * (u_size + 1) / 2.0)
        self.l_idx = list(range(self.lower_size))
        super(A, self).__init__(**kwargs)

        # Some precalculating for call()
        self.diag_idx = list(range(u_size))
        self.lower_idx1 = []
        self.lower_idx2 = []
        for i in self.diag_idx:
            for j in range(i):
                self.lower_idx1.append(i)
                self.lower_idx2.append(j)

    def build(self, input_shape):
        super(A, self).build(self.lower_size + self.u_size)
        
    def get_output_shape_for(self, input_shape):
        return (None, 1)
        
    def _p(self, x):
        l = T.zeros((x.shape[0], self.u_size, self.u_size))
        l = T.set_subtensor(l[:, self.diag_idx, self.diag_idx], T.exp(x[:, self.diag_idx]))
        if self.u_size > 1:
            l = T.set_subtensor(
                l[:, self.lower_idx1, self.lower_idx2],
                x[:, self.u_size:self.u_size + self.u_size]
            )
        return K.batch_dot(l, K.permute_dimensions(l, [0, 2, 1]))
    
    def call(self, x, mask=None):
        p = self._p(x[:, :self.lower_size])
        u = x[:, self.lower_size:self.lower_size + self.u_size]
        mu = x[:, self.lower_size + self.u_size:]
        d = K.expand_dims(u - mu, -1)
        a = -T.batched_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)
        return a.reshape((x.shape[0], 1))


class NNet():
    
    def __init__(self, x_size, u_size, mu_scaling, hidden_size=100):
        self.x_size = x_size
        self.u_size = u_size
        self.hidden_size = hidden_size
    
        self.x = Input(shape=(self.x_size, ), name='x')
        fc1 = Dense(input_dim=self.x_size, output_dim=self.hidden_size, activation='relu', name='fc1')(self.x)
        fc2 = Dense(input_dim=self.hidden_size, output_dim=self.hidden_size, activation='relu', name='fc2')(fc1)
        
        v = Dense(input_dim=self.hidden_size, output_dim=1, name='v')(fc2)
        self.v = Model(input=self.x, output=v)
        self.v.build(input_shape=(self.x_size, ))
        
        mu = Dense(input_dim=self.hidden_size, output_dim=self.u_size, activation='tanh', name='mu_dense')(fc2)
        mu_scaled = Lambda(lambda x: mu_scaling * x)(mu)
        self.mu = Model(input=self.x, output=mu_scaled)
        self.mu.build(input_shape=(self.x_size, ))
        
        l_all = Dense(
            input_dim=self.hidden_size,
            output_dim=int(self.u_size * (self.u_size + 1) / 2.0)
        )(fc2)
        
        self.l_all = Model(input=self.x, output=l_all)
        
        u = Input(shape=(self.u_size, ), name='u')
        a = A(u_size=self.u_size, name='A')
        
        
        self.u = Model(input=u, output=u, name='u_model')
        self.a = Sequential(
            layers=[
                Merge([self.l_all, self.u, self.mu], mode='concat', name='merge_for_p'),
                a
            ]
        )
        
        self.q = Sequential(layers=[Merge([self.a, self.v])])
        adam = Adam(lr=0.0001)
        self.q.compile(loss='mse', optimizer=adam)
