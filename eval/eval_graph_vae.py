import os
import sys
import math
import pickle
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
import numpy as np
from transformers import get_scheduler
from model.box_vae import SceneVAEModel

class Evaluator(object):
    def __init__(self, args=None, logger=None, eval_dataloader=None, vocab=None, device=None):
        # Checkpoint and log setting
        self.args = args
        self.logger = logger
        self.device = device

        # DataLoader
        self.vocab = vocab
        self.eval_dataloader = eval_dataloader

        # Model setting
        num_objs = len(self.vocab['object_idx_to_name'])
        num_rels = len(self.vocab['pred_idx_to_name'])
        self.vae = SceneVAEModel(self.args, num_objs, num_rels)
        self.vae.to(self.device)
        self.vae.eval()

        self.save_path = self.args.output_dir

    def start(self, save_path=None):
        # Exit if path do not exist
        ckpt_path = os.path.join(save_path, 'ckpt.pt')
        try:
            if (not os.path.exists(ckpt_path)): raise ValueError
        except ValueError :
            print('Checkpoint is not exist!')
            sys.exit(0)
        
        stats_path = os.path.join(save_path, 'stats.pkl')
        stats = pickle.load(open(stats_path, 'rb'))
        self.mean_est, self.cov_est = stats[0], stats[1]

        # Load model parameter
        self.save_path = os.path.join(save_path, 'layout')
        os.mkdir(self.save_path)
        self.vae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.logger.info('Evaluating the model at {}'.format(ckpt_path))

        # Eval
        self.validate()

    def validate(self):
        self.logger.info('Starting evaluation')

        with torch.no_grad():
            pbar = tqdm(self.eval_dataloader, file=sys.stdout)
            for idx, batch in enumerate(pbar):
                
                imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
                objs, triples = objs.to(self.device), triples.to(self.device)
                obj_clip_embs, rel_clip_embs = obj_clip_embs.to(self.device), rel_clip_embs.to(self.device)
                box_preds = self.vae.sample_box(self.mean_est, self.cov_est, objs, obj_clip_embs, triples, rel_clip_embs, self.device)

                objs = objs.cpu().detach().numpy()
                box_preds = box_preds.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                self.save_sample(objs, box_preds, boxes, img_paths[0])

    def save_sample(self, objs, boxes_pred, boxes_gt, img_paths):
        name = os.path.basename(img_paths)
        save_path = os.path.join(self.save_path, name)
        color = list(np.random.choice(range(256), size=(len(boxes_pred), 3)))

        layout_pred = Image.new('RGB', size=(self.args.resolution, self.args.resolution)) # Shape: W, H
        draw_pred = ImageDraw.Draw(layout_pred)
        for i, (obj, box_pred) in enumerate(zip(objs, boxes_pred)):
            obj_text = self.vocab['object_idx_to_name'][obj]
            box_pred = box_pred * self.args.resolution
            x0, y0, x1, y1 = box_pred 
            if x1 < x0 or y1 < y0:
                break
            draw_pred.rectangle([x0, y0, x1, y1], outline=tuple(color[i]))
            draw_pred.text(xy=(x0, y0), text=obj_text, fill=tuple(color[i]))
        
        layout_gt = Image.new('RGB', size=(self.args.resolution, self.args.resolution)) # Shape: W, H
        draw_gt = ImageDraw.Draw(layout_gt)
        for i, (obj, box_gt) in enumerate(zip(objs, boxes_gt)):
            obj_text = self.vocab['object_idx_to_name'][obj]
            box_gt = box_gt * self.args.resolution
            x0, y0, x1, y1 = box_gt 
            draw_gt.rectangle([x0, y0, x1, y1], outline=tuple(color[i]))
            draw_gt.text(xy=(x0, y0), text=obj_text, fill=tuple(color[i]))
        
        layout = Image.new('RGB', size=(self.args.resolution + self.args.resolution, self.args.resolution))
        layout.paste(layout_gt, (0, 0))
        layout.paste(layout_pred, (self.args.resolution, 0))
        layout.save(save_path)
