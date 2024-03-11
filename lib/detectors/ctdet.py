from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import TRUE

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch


from lib.external.nms import soft_nms

  #print('NMS not imported! If you need it,'
        #' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      # print(output)
      hm_p = output['hm_p'].sigmoid_()
      wh_p = output['wh_p']
      reg_p = output['reg_p'] if self.opt.reg_offset else None
      hm_h = output['hm_h'].sigmoid_()
      wh_h = output['wh_h']
      reg_h = output['reg_h'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm_p = (hm_p[0:1] + flip_tensor(hm_p[1:2])) / 2
        wh_p = (wh_p[0:1] + flip_tensor(wh_p[1:2])) / 2
        reg_p = reg_p[0:1] if reg_p is not None else None
        hm_h = (hm_h[0:1] + flip_tensor(hm_h[1:2])) / 2
        wh_h = (wh_h[0:1] + flip_tensor(wh_h[1:2])) / 2
        reg_h = reg_h[0:1] if reg_h is not None else None
      # print('hm_p:', hm_p.shape)
      torch.cuda.synchronize()
      forward_time = time.time()
      dets_p, dets_h = ctdet_decode(hm_p, wh_p, hm_h, wh_h, reg_p=reg_p, reg_h= reg_h, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      # print('dets_p:', dets_p)
    if return_time:
      return output, dets_p, dets_h, forward_time
    else:
      return output, dets_p, dets_h

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets_p, dets_h, output, scale=1):
    detection_p = dets_p.detach().cpu().numpy().copy()
    detection_p[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm_p'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets_p[i])):
        if detection_p[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection_p[i, k, :4], detection_p[i, k, -1],
                                 detection_p[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))
    detection_h = dets_h.detach().cpu().numpy().copy()
    detection_h[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm_h'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets_p[i])):
        if detection_h[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection_h[i, k, :4], detection_h[i, k, -1],
                                 detection_h[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def joint_NMS(self, results):
    def calculateIOU(box1, box2):
        x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
        xx1, yy1, xx2, yy2 = box2[0], box2[1], box2[2], box2[3]
        w = max(0, min(x2, xx2) - max(x1, xx1))
        h = max(0, min(y2, yy2) - max(y1, yy1))
        interarea = w * h
        union = (x2 - x1) * (y2 - y1) + (xx2 - xx1) * (yy2 - yy1) - interarea
        IOU = interarea / union
        return IOU

    head_boxes = []
    person_boxes = []
    p_results = []


    for j in range(1, self.num_classes + 1):
        for bbox in results[j]:
            if bbox[4] > self.opt.vis_thresh:
                if j == 2:
                    head_boxes.append(bbox)
                elif j == 1:
                    person_boxes.append(bbox)

    for pbox in person_boxes:
        max_score = float('-inf')
        best_head_box = None
        head_region = [pbox[0], pbox[1], pbox[2], pbox[1] + 0.33 * (pbox[3] - pbox[1])]

        for hbox in head_boxes:
            if hbox[0] >= head_region[0] and hbox[1] >= head_region[1] and hbox[2] <= head_region[2] and hbox[3] <= head_region[3]:
                if hbox[4] > max_score:
                    max_score = hbox[4]
                    best_head_box = hbox

        if best_head_box is not None:
            p_results.append(pbox)


    final_results = []
    final_results_1 = {}
    while len(person_boxes) > 0:
        person_boxes.sort(key=lambda x: x[4], reverse=True)
        max_box = person_boxes.pop(0)
        final_results.append(max_box)

        remove_indices = []
        for i, other_box in enumerate(person_boxes):
            iou = calculateIOU(max_box, other_box)
            if iou > 0.5 and not any(np.array_equal(other_box, x) for x in p_results):
                remove_indices.append(i)

        for i in reversed(remove_indices):
            person_boxes.pop(i)
            final_results[1] = final_results_1
            final_results[2] = head_boxes
    return final_results

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    person_bbox_count = 0
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          if j == 1:
            person_bbox_count += 1

          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    txt = 'person_bbox_count{}'.format(person_bbox_count)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debugger.imgs['ctdet'], txt, (10, 30), 
                 font, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    debugger.show_all_imgs(pause=self.pause)

