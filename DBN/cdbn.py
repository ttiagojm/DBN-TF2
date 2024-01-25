import tensorflow as tf
from tqdm import tqdm

class CDBN(tf.Module):

    def __init__(self, batch_size, global_step = 1, **kwargs):
        super(CDBN, self).__init__(**kwargs)
        self.global_step = global_step
        self.batch_size = batch_size
        self.crbms = list()

    def pretrain(self, dataset):

        data = dataset

        for crbm in self.crbms:
            for i in tqdm(range(self.global_step)):
                for batch in data:
                    crbm.do_contrastive_divergence(batch, global_step=i)

            data = data.map(lambda x: crbm.infer_probability(x, "forward"))

    def encode(self, data):
        for crbm in self.crbms:
            h = crbm.infer_probability(data, "forward")
            data = h
        return h

    def decode(self, data):
        for crbm in self.crbms[::-1]:
            data = crbm.infer_probability(data, "backward")

        return data