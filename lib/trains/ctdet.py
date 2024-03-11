from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss_p, hm_loss_h, wh_loss_p, wh_loss_h, off_loss_p, off_loss_h = 0, 0, 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      #print('output:', output)
      if not opt.mse_loss:
        output['hm_p'] = _sigmoid(output['hm_p'])
        output['hm_h'] = _sigmoid(output['hm_h'])
      if opt.eval_oracle_hm:
        output['hm_p'] = batch['hm_p']
        output['hm_h'] = batch['hm_h']
      if opt.eval_oracle_wh:
        output['wh_p'] = torch.from_numpy(gen_oracle_map(
          batch['wh_p'].detach().cpu().numpy(), 
          batch['ind_p'].detach().cpu().numpy(), 
          output['wh_p'].shape[3], output['wh_p'].shape[2])).to(opt.device)
        output['wh_h'] = torch.from_numpy(gen_oracle_map(
          batch['wh_h'].detach().cpu().numpy(), 
          batch['ind_h'].detach().cpu().numpy(), 
          output['wh_h'].shape[3], output['wh_h'].shape[2])).to(opt.device)  
      if opt.eval_oracle_offset:
        output['reg_p'] = torch.from_numpy(gen_oracle_map(
          batch['reg_p'].detach().cpu().numpy(), 
          batch['ind_p'].detach().cpu().numpy(), 
          output['reg_p'].shape[3], output['reg_p'].shape[2])).to(opt.device)
        output['reg_h'] = torch.from_numpy(gen_oracle_map(
          batch['reg_h'].detach().cpu().numpy(), 
          batch['ind_h'].detach().cpu().numpy(), 
          output['reg_h'].shape[3], output['reg_h'].shape[2])).to(opt.device)

      hm_loss_p += self.crit(output['hm_p'], batch['hm_p']) / opt.num_stacks
      hm_loss_h += self.crit(output['hm_h'], batch['hm_h']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight_p = batch['dense_wh_mask_p'].sum() + 1e-4
          mask_weight_h = batch['dense_wh_mask_h'].sum() + 1e-4
          wh_loss_p += (
            self.crit_wh(output['wh_p'] * batch['dense_wh_mask_p'],
            batch['dense_wh_p'] * batch['dense_wh_mask_p']) / 
            mask_weight_p) / opt.num_stacks
          wh_loss_h += (
            self.crit_wh(output['wh_h'] * batch['dense_wh_mask_h'],
            batch['dense_wh_h'] * batch['dense_wh_mask_h']) / 
            mask_weight_h) / opt.num_stacks  
        elif opt.cat_spec_wh:
          wh_loss_p += self.crit_wh(
            output['wh_p'], batch['cat_spec_mask_p'],
            batch['ind_p'], batch['cat_spec_wh_p']) / opt.num_stacks
          wh_loss_h += self.crit_wh(
            output['wh_h'], batch['cat_spec_mask_h'],
            batch['ind_h'], batch['cat_spec_wh_h']) / opt.num_stacks  
        else:
          wh_loss_p += self.crit_reg(
            output['wh_p'], batch['reg_mask_p'],
            batch['ind_p'], batch['wh_p']) / opt.num_stacks
          wh_loss_h += self.crit_reg(
            output['wh_h'], batch['reg_mask_h'],
            batch['ind_h'], batch['wh_h']) / opt.num_stacks

      if opt.reg_offset and opt.off_weight > 0:
        off_loss_p += self.crit_reg(output['reg_p'], batch['reg_mask_p'],
                             batch['ind_p'], batch['reg_p']) / opt.num_stacks
        off_loss_h += self.crit_reg(output['reg_h'], batch['reg_mask_h'],
                             batch['ind_h'], batch['reg_h']) / opt.num_stacks
    loss_p = opt.hm_weight * hm_loss_p + opt.wh_weight * wh_loss_p + \
           opt.off_weight * off_loss_p
    loss_h = opt.hm_weight * hm_loss_h + opt.wh_weight * wh_loss_h + \
           opt.off_weight * off_loss_h
    loss = 0.5 * loss_p + 0.5 * loss_h
    loss_stats = {'loss': loss, 'loss_p': loss_p, 'hm_loss_p': hm_loss_p,
                  'wh_loss_p': wh_loss_p, 'off_loss_p': off_loss_p, 'loss_h': loss_h,
                  'hm_loss_h': hm_loss_h, 'wh_loss_h': wh_loss_h, 'off_loss_h': off_loss_h}
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'loss_p', 'hm_loss_p', 'wh_loss_p', 'off_loss_p',
                    'loss_h', 'hm_loss_h', 'wh_loss_h', 'off_loss_h']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]