import os
import h5py  
import json 
import random
import pickle
import argparse
import collections
from PIL import Image

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class VisualGenomeDataset(Dataset):
    def __init__(self, args, vocab, mode, tokenizer=None):
        super(VisualGenomeDataset, self).__init__()
        assert mode in ["train", "val", "test"]

        self.mode = mode
        self.image_dir = os.path.join(args.data_dir, 'images')
        self.resolution = args.resolution

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.num_objects = len(self.vocab['object_idx_to_name'])

        self.labels = {}
        self.image_paths = []
        with h5py.File(os.path.join(args.data_dir, 'labels', f'{self.mode}.h5'), 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.labels[k] = torch.IntTensor(np.asarray(v))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.labels['object_names'])
    
    def __getitem__(self, index):
        image_path = self.image_paths[index].decode("utf-8")
        image_path = os.path.join(self.image_dir, image_path)

        image = Image.open(image_path).convert("RGB")
        WW, HH = image.size
        image = self.image_transforms(image)
        clip_embs =  pickle.load(open(image_path.replace('images', 'clip').replace('.jpg', '.pkl'), 'rb'))
        clip_obj_embs = torch.from_numpy(clip_embs['objects'])
        clip_rel_embs = torch.from_numpy(clip_embs['relations'])
        
        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.labels['objects_per_image'][index].item()))
        for rel_idx in range(self.labels['relationships_per_image'][index].item()):
            s = self.labels['relationship_subjects'][index, rel_idx].item()
            o = self.labels['relationship_objects'][index, rel_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        obj_idxs += obj_idxs_without_rels
        
        objs = []
        boxes = []
        words = []
        obj_idx_mapping = {}
        counter = 0
        for i, obj_idx in enumerate(obj_idxs):
            curr_obj = self.labels['object_names'][index, obj_idx].item()
            x, y, w, h = self.labels['object_boxes'][index, obj_idx].tolist()

            x0, x1 = float(x) / WW, float(x + w) / WW
            y0, y1 = float(y) / HH, float(y + h) / HH
            curr_box = (torch.FloatTensor([x0, y0, x1, y1]))
            words.append(self.vocab['object_idx_to_name'][curr_obj])

            objs.append(curr_obj)
            boxes.append(curr_box)
            obj_idx_mapping[obj_idx] = counter
            counter += 1

        # The last object will be the special __image__ object
        objs.append(self.vocab['object_name_to_idx']['__image__'])
        boxes.append(torch.FloatTensor([0, 0, 1, 1]))

        boxes = torch.stack(boxes)
        objs = torch.LongTensor(objs)
        num_objs = counter + 1

        triples = []
        for rel_idx in range(self.labels['relationships_per_image'][index].item()):
            s = self.labels['relationship_subjects'][index, rel_idx].item()
            p = self.labels['relationship_predicates'][index, rel_idx].item()
            o = self.labels['relationship_objects'][index, rel_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        caption = ''
        for word in words:  
            text = word + '; '               
            caption += text
        caption = caption[:-2]

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(len(objs) - 1):
            triples.append([i, in_image, num_objs - 1])
        triples = torch.LongTensor(triples)

        caption = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return image, objs, clip_obj_embs, boxes, triples, clip_rel_embs, image_path, caption
    

def collate_fn_graph_batch(batch):
  """
  Collate function to be used when wrapping a VisualGenomeDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, 3, H, W)
  - objs: LongTensor of shape (num_objs,) giving categories for all objects
  - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
  - triples: FloatTensor of shape (num_triples, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n]
  - imgs_masked: FloatTensor of shape (N, 4, H, W)
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_objs, all_clip_obj_embs, all_boxes, all_triples, all_clip_rel_embs, all_img_paths, all_captions = [], [], [], [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []

  obj_offset = 0
  for i, (img, objs, clip_obj_emb, boxes, triples, clip_rel_embs, img_path, caption) in enumerate(batch):

    num_objs, num_triples = objs.size(0), triples.size(0)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    obj_offset += num_objs

    all_imgs.append(img[None])
    all_objs.append(objs)
    all_clip_obj_embs.append(clip_obj_emb)
    all_boxes.append(boxes)
    all_triples.append(triples)
    all_clip_rel_embs.append(clip_rel_embs)
    all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
    all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))
    all_img_paths.append(img_path)
    all_captions.append(caption)

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_clip_obj_embs = torch.cat(all_clip_obj_embs)
  all_boxes = torch.cat(all_boxes)
  all_triples = torch.cat(all_triples)
  all_clip_rel_embs = torch.cat(all_clip_rel_embs)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)
  all_captions = torch.cat(all_captions)

  return all_imgs, all_objs, all_clip_obj_embs, all_boxes, all_triples, all_clip_rel_embs, all_obj_to_img, all_triple_to_img, all_img_paths, all_captions

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a dataloader.")
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--data_dir', type=str, default='/data/anonymity/VisualGenome', help='path to training dataset')
    parser.add_argument('--output_dir', type=str, default='', help='path to save checkpoint')
    parser.add_argument('--resolution', type=int, default=512, help='resolution')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--dataloader_shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    
    args = parser.parse_args()
    return args

def build_train_dataloader(args, tokenizer=None):

    with open(os.path.join(args.data_dir, 'vocab.json'), 'r') as f:
        vocab = json.load(f)

    train_dataset = VisualGenomeDataset(args, vocab=vocab, mode='train', tokenizer=tokenizer)
    iter_per_epoch = len(train_dataset) // args.batch_size

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=args.dataloader_shuffle,
        collate_fn=collate_fn_graph_batch
    )

    val_dataset = VisualGenomeDataset(args, vocab=vocab, mode='val', tokenizer=tokenizer)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        collate_fn=collate_fn_graph_batch
    )

    return train_dataloader, val_dataloader, vocab

if __name__ == '__main__':
    args = parse_args()
    train_loader, val_loader = build_train_dataloader(args=args)
    dataloader = iter(train_loader)
    from  tqdm import tqdm
    import sys
    pbar = tqdm(range(100), file=sys.stdout)
    for idx in pbar:
        all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img = next(dataloader)
        print(all_triples.shape)






