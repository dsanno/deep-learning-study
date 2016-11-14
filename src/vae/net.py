import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class VAE(chainer.Chain):

    def __init__(self, hidden_size=500, latent_size=30):
        super(VAE, self).__init__(
            enc1 = L.Linear(28 * 28, hidden_size),
            enc2 = L.Linear(hidden_size, hidden_size),
            enc_mean = L.Linear(hidden_size, latent_size),
            enc_var  = L.Linear(hidden_size, latent_size),
            dec1 = L.Linear(latent_size, hidden_size),
            dec2 = L.Linear(hidden_size, hidden_size),
            dec3 = L.Linear(hidden_size, 28 * 28)
        )

    def __call__(self, x, train=True):
        xp = self.xp
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        mean = self.enc_mean(h2)
        var  = 0.5 * self.enc_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(mean.dtype)
        z  = mean + F.exp(var) * rand
        g1 = F.relu(self.dec1(z))
        g2 = F.relu(self.dec2(g1))
        g3 = F.sigmoid(self.dec3(g2))
        return (g3, mean, var)

    def generate(self, z):
        g1 = F.relu(self.dec1(z))
        g2 = F.relu(self.dec2(g1))
        return F.sigmoid(self.dec3(g2))

class VAEM2(chainer.Chain):

    def __init__(self, hidden_size=500, latent_size=30, class_size=10):
        super(VAEM2, self).__init__(
            enc_embed=L.EmbedID(class_size, hidden_size),
            enc1=L.Linear(28 * 28, hidden_size),
            enc2=L.Linear(hidden_size, hidden_size),
            enc_mean=L.Linear(hidden_size, latent_size),
            enc_var=L.Linear(hidden_size, latent_size),
            dec_embed=L.EmbedID(class_size, hidden_size),
            dec1=L.Linear(latent_size, hidden_size),
            dec2=L.Linear(hidden_size, hidden_size),
            dec3=L.Linear(hidden_size, 28 * 28)
        )

    def __call__(self, x, t, gen_t, train=True):
        xp = self.xp
        h1 = F.relu(self.enc1(x) + self.enc_embed(t))
        h2 = F.relu(self.enc2(h1))
        mean = self.enc_mean(h2)
        var  = 0.5 * self.enc_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(mean.dtype)
        z  = mean + F.exp(var) * rand
        g1 = F.relu(self.dec1(z) + self.dec_embed(gen_t))
        g2 = F.relu(self.dec2(g1))
        g3 = F.sigmoid(self.dec3(g2))
        return (g3, mean, var)

    def generate(self, z, t):
        g1 = F.relu(self.dec1(z) + self.dec_embed(t))
        g2 = F.relu(self.dec2(g1))
        return F.sigmoid(self.dec3(g2))
