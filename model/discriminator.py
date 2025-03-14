import torch.nn as nn
import torch
import torch.nn.functional as F

def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

class BoxDiscriminator(nn.Module):
    """
    Relationship discriminator based on bounding boxes. For a given object pair, it takes their
    semantic labels, the relationship label and the two bounding boxes of the pair and judges
    whether this is a plausible occurence.
    """
    def __init__(self, box_dim, rel_dim, obj_dim):
        super(BoxDiscriminator, self).__init__()

        self.rel_dim = rel_dim
        self.obj_dim = obj_dim
        in_size = box_dim * 2 + rel_dim + obj_dim * 2

        self.D = nn.Sequential(nn.Linear(in_size, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 1),
                               nn.Sigmoid())

        self.D.apply(_init_weights)

    def forward(self, objs, triples, boxes, with_grad=False, is_real=False):

        s_idx, predicates, o_idx = triples.chunk(3, dim=1)
        predicates = predicates.squeeze(1)
        s_idx = s_idx.squeeze(1)
        o_idx = o_idx.squeeze(1)
        subjectBox = boxes[s_idx]
        objectBox = boxes[o_idx]
        predicates = to_one_hot_vector(self.rel_dim, predicates)

        subjectCat = to_one_hot_vector(self.obj_dim, objs[s_idx])
        objectCat = to_one_hot_vector(self.obj_dim, objs[o_idx])
        x = torch.cat([subjectCat, objectCat, predicates, subjectBox, objectBox], 1)


        reg = None
        if with_grad:
            x.requires_grad = True
            y = self.D(x)
            reg = discriminator_regularizer(y, x, is_real)
            x.requires_grad = False
        else:
            y = self.D(x)
        
        return y, reg

def discriminator_regularizer(logits, arg, is_real):

    logits.backward(torch.ones_like(logits), retain_graph=True)
    grad_logits = arg.grad
    grad_logits_norm = torch.norm(grad_logits, dim=1).unsqueeze(1)

    assert grad_logits_norm.shape == logits.shape

    # tf.multiply -> element-wise mul
    if is_real:
        reg = (1.0 - logits)**2 * (grad_logits_norm)**2
    else:
        reg = (logits)**2 * (grad_logits_norm)**2

    return reg


def to_one_hot_vector(num_class, label):
    """ Converts a label to a one hot vector

    :param num_class: number of object classes
    :param label: integer label values
    :return: a vector of the length num_class containing a 1 at position label, and 0 otherwise
    """
    return torch.nn.functional.one_hot(label, num_class).float()