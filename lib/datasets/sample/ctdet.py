from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

        
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm_p = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
    hm_h = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh_p = np.zeros((self.max_objs, 2), dtype=np.float32)
    wh_h = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh_p = np.zeros((2, output_h, output_w), dtype=np.float32)
    dense_wh_h = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg_p = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg_h = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind_p = np.zeros((self.max_objs), dtype=np.int64)
    ind_h = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask_p = np.zeros((self.max_objs), dtype=np.uint8)
    reg_mask_h = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh_p = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_wh_h = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask_p = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    cat_spec_mask_h = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id == 0:
        bbox_p = self._coco_box_to_bbox(ann['bbox'])
        
        if flipped:
          bbox_p[[0, 2]] = width - bbox_p[[2, 0]] - 1
        bbox_p[:2] = affine_transform(bbox_p[:2], trans_output)
        bbox_p[2:] = affine_transform(bbox_p[2:], trans_output)
        bbox_p[[0, 2]] = np.clip(bbox_p[[0, 2]], 0, output_w - 1)
        bbox_p[[1, 3]] = np.clip(bbox_p[[1, 3]], 0, output_h - 1)
        h_p, w_p = bbox_p[3] - bbox_p[1], bbox_p[2] - bbox_p[0]
        if h_p > 0 and w_p > 0:
          radius_p = gaussian_radius((math.ceil(h_p), math.ceil(w_p)))
          radius_p = max(0, int(radius_p))
          radius_p = self.opt.hm_gauss if self.opt.mse_loss else radius_p
          ct_p = np.array(
            [(bbox_p[0] + bbox_p[2]) / 2, (bbox_p[1] + bbox_p[3]) / 2], dtype=np.float32)
          ct_int_p = ct_p.astype(np.int32)
          draw_gaussian(hm_p[cls_id], ct_int_p, radius_p)
          wh_p[k] = 1. * w_p, 1. * h_p
          ind_p[k] = ct_int_p[1] * output_w + ct_int_p[0]
          reg_p[k] = ct_p - ct_int_p
          reg_mask_p[k] = 1
          cat_spec_wh_p[k, cls_id * 2: cls_id * 2 + 2] = wh_p[k]
          cat_spec_mask_p[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh_p, hm_p.max(axis=0), ct_int_p, wh_p[k], radius_p)
          gt_det.append([ct_p[0] - w_p / 2, ct_p[1] - h_p / 2, 
                        ct_p[0] + w_p / 2, ct_p[1] + h_p / 2, 1, cls_id])
      else:
        bbox_h = self._coco_box_to_bbox(ann['bbox'])
        
        if flipped:
          bbox_h[[0, 2]] = width - bbox_h[[2, 0]] - 1
        bbox_h[:2] = affine_transform(bbox_h[:2], trans_output)
        bbox_h[2:] = affine_transform(bbox_h[2:], trans_output)
        bbox_h[[0, 2]] = np.clip(bbox_h[[0, 2]], 0, output_w - 1)
        bbox_h[[1, 3]] = np.clip(bbox_h[[1, 3]], 0, output_h - 1)
        h, w = bbox_h[3] - bbox_h[1], bbox_h[2] - bbox_h[0]
        if h > 0 and w > 0:
          radius_h = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius_h = max(0, int(radius_h))
          radius_h = self.opt.hm_gauss if self.opt.mse_loss else radius_h
          ct_h = np.array(
            [(bbox_h[0] + bbox_h[2]) / 2, (bbox_h[1] + bbox_h[3]) / 2], dtype=np.float32)
          ct_int_h = ct_h.astype(np.int32)
          draw_gaussian(hm_h[cls_id], ct_int_h, radius_h)
          wh_h[k] = 1. * w, 1. * h
          ind_h[k] = ct_int_h[1] * output_w + ct_int_h[0]
          reg_h[k] = ct_h - ct_int_h
          reg_mask_h[k] = 1
          cat_spec_wh_h[k, cls_id * 2: cls_id * 2 + 2] = wh_h[k]
          cat_spec_mask_h[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh_h, hm_h.max(axis=0), ct_int_h, wh_h[k], radius_h)
          gt_det.append([ct_h[0] - w / 2, ct_h[1] - h / 2, 
                        ct_h[0] + w / 2, ct_h[1] + h / 2, 1, cls_id])
    ret = {'input': inp, 'hm_p': hm_p, 'reg_mask_p': reg_mask_p, 'ind_p': ind_p, 'wh_p': wh_p, 
            'hm_h': hm_h, 'reg_mask_h': reg_mask_h, 'ind_h': ind_h, 'wh_h': wh_h}
    if self.opt.dense_wh:
      hm_a = hm_h.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh_h, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh_h, 'cat_spec_mask': cat_spec_mask_h})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg_p': reg_p, 'reg_h': reg_h})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret