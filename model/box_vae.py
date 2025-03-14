import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from .gcn import GraphTripleConvNet, _init_weights, build_mlp

class SceneVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a separate embedding of shape and bounding box latents.
    """
    def __init__(self, args, num_objs, num_rels):
        super(SceneVAEModel, self).__init__()

        gconv_dim = args.embedding_dim          # 64
        gconv_hidden_dim = gconv_dim * 4        # 64 * 4
        box_embedding_dim = args.embedding_dim  # 64
        obj_embedding_dim = args.embedding_dim  # 64


        self.obj_embeddings_encoder = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.obj_embeddings_decoder = nn.Embedding(num_objs + 1, obj_embedding_dim)

        self.rel_embeddings_encoder = nn.Embedding(num_rels, args.embedding_dim * 2)
        self.rel_embeddings_decoder = nn.Embedding(num_rels, args.embedding_dim * 2)

        self.box_embeddings = nn.Linear(4, box_embedding_dim)

        self.mlp_mean_var = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, args.embedding_dim * 2], 
            batch_norm="batch",
            final_nonlinearity=True
        )
        self.mlp_mean = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim], 
            batch_norm="batch", 
            final_nonlinearity=False
        )
        self.mlp_var = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim], 
            batch_norm="batch",  
            final_nonlinearity=False
        )
        self.mlp_box = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, 4], 
            batch_norm="batch", 
            final_nonlinearity=False
        )
        # self.mlp_box = build_mlp(
        #     [args.embedding_dim * 2 + 512, gconv_hidden_dim, 4], 
        #     batch_norm="batch", 
        #     final_nonlinearity=True
        # )

        gconv_encoder_kwargs = {
            'input_dim_obj':        gconv_dim * 2 + 512,
            'input_dim_pred':       gconv_dim * 2 + 512,
            'hidden_dim':           gconv_hidden_dim,
            'num_layers':           5,
            'pooling':              'avg',
            'mlp_normalization':    'batch',
            'residual':             True#
        }

        gconv_decoder_kwargs = {
            'input_dim_obj':        gconv_dim * 2 + 512,
            'input_dim_pred':       gconv_dim * 2 + 512,
            'hidden_dim':           gconv_hidden_dim,
            'num_layers':           5,
            'pooling':              'avg',
            'mlp_normalization':    'batch',
            'residual':             True
        }

        self.gconv_encoder = GraphTripleConvNet(**gconv_encoder_kwargs)
        self.gconv_decoder = GraphTripleConvNet(**gconv_decoder_kwargs)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.mlp_mean_var.apply(_init_weights)
        self.mlp_mean.apply(_init_weights)
        self.mlp_var.apply(_init_weights)
        self.mlp_box.apply(_init_weights)

    def encoder(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs):
        ##print(list(self.mlp_mean.parameters()))
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)               # Shape: (T, 1), s-subject, p-predicate (relation), o-object
        s, p, o = [x.squeeze(1) for x in [s, p, o]]     # Shape: (T,)
        edges = torch.stack([s, o], dim=1)              # Shape: (T, 2)

        #Relation Embeding
        rel_embs = self.rel_embeddings_encoder(p)                 # Shape: (T, embedding_dim * 2) = (T, 64 * 2)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)    # Shape: (T, clip_dim + embedding_dim * 2) = (T, 512 + 64 * 2)

        # Node Embeding
        obj_embs = self.obj_embeddings_encoder(objs)                # Shape: (O, embedding_dim) = (O, 64)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)      # Shape: (O, clip_dim + embedding_dim) = (O, 512 + 64)
        box_embs = self.box_embeddings(boxes)                       # Shape: (O, embedding_dim) = (O, 64)
        obj_embs = torch.cat([obj_embs, box_embs], dim=1)           # Shape: (O, clip_dim + embedding_dim * 2) = (O, 512 + 64 * 2)
        
        # Encoding
        all_embs, _ = self.gconv_encoder(obj_embs, rel_embs, edges)     # Shape: (O, clip_dim + embedding_dim * 2) = (O, 512 + 64 * 2)
        all_embs = self.mlp_mean_var(all_embs)                          # Shape: (O, embedding_dim * 2) = (O, 64 * 2)
        mu = self.mlp_mean(all_embs)                                    # Shape: (O, embedding_dim) = (O, 64)
        logvar = self.mlp_var(all_embs)                                 # Shape: (O, embedding_dim) = (O, 64)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder(self, objs, obj_clip_embs, z, triples, rel_clip_embs):

        s, p, o = triples.chunk(3, dim=1)               # Shape: (T, 1), s-subject, p-predicate, o-object
        s, p, o = [x.squeeze(1) for x in [s, p, o]]     # Shape: (T,)
        edges = torch.stack([s, o], dim=1)              # Shape: (T, 2)

        #Relation Embeding
        rel_embs = self.rel_embeddings_decoder(p)                   # Shape: (T, embedding_dim) = (T, embedding_dim)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)      # Shape: (T, clip_dim + embedding_dim) = (T, 512 + 64)

        # Node Embeding 
        obj_embs = self.obj_embeddings_decoder(objs)                       # Shape: (O, embedding_dim) = (O, 64)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)    # Shape: (O, clip_dim + embedding_dim) = (T, 512 + 64)
        obj_embs = torch.cat([obj_embs, z], dim=1)                # Shape: (O, clip_dim + embedding_dim * 2) = (T, 512 + 64 * 2)

        # Decoding
        all_embs, _ = self.gconv_decoder(obj_embs, rel_embs, edges)
        box_pred = self.mlp_box(all_embs)

        return torch.sigmoid(box_pred)
    
    def forward(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs):
        mu, logvar = self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)

        z = self.reparameterize(mu, logvar)

        box_pred = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs)
        return mu, logvar, box_pred

    def sample_box(self, mean_est, cov_est, objs, obj_clip_embs, triples, rel_clip_embs, device):
        with torch.no_grad():
            # mean_est = np.zeros(64)
            # cov_est = np.eye(64)
            z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().to(device)
            box_pred = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs)

            return box_pred
    
    def collect_data_statistics(self, train_loader, device):
        
        pbar = tqdm(train_loader, file=sys.stdout)
        mean_cat = []
        for idx, batch in enumerate(pbar):
        #for idx, batch in enumerate(train_loader):

            imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
            objs, triples, boxes = objs.to(device), triples.to(device), boxes.to(device)
            obj_clip_embs, rel_clip_embs = obj_clip_embs.to(device), rel_clip_embs.to(device)

            mean, logvar = self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
            mean, logvar = mean.cpu().clone(), logvar.cpu().clone()

            mean = mean.data.cpu().clone()
            mean_cat.append(mean)

        mean_cat = torch.cat(mean_cat, dim=0)
        mean_est = torch.mean(mean_cat, dim=0, keepdim=True)   # Shape: (1, embedding_dim) = (1, 64)
        cov_est = np.cov((mean_cat - mean_est).numpy().T)
        mean_est = mean_est[0]

        return mean_est, cov_est
