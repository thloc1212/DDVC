# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys

import os
import json
import time
import torch
import sys
import collections
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath

ddvc_dir = dirname(abspath(__file__))
sys.path.insert(0, ddvc_dir)
sys.path.insert(0, os.path.join(ddvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(ddvc_dir, 'densevid_eval3/SODA'))

from eval_utils_clip import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from data.video_dataset import PropSeqDataset, collate_fn
from ddvc.ddvc_ret_encdec_clip_capQ import build
from collections import OrderedDict
from ret_utils import load_clip_memory_bank, load_clip_gt
from misc.training_state import (atomic_save, capture_rng, restore_rng, snapshot,
                                 schedule_contrastive, validate_resume, plain_metadata)
from misc.runtime_perf import amp_context,configure_runtime,vram_metrics,append_performance

# global clip_gt
def train(opt):
    set_seed(opt.seed)
    runtime=configure_runtime(opt)
    accumulation = getattr(opt, 'gradient_accumulation_steps', 1)
    log_every = getattr(opt, 'progress_log_every', 10)
    if accumulation < 1 or log_every < 1:
        raise ValueError('gradient_accumulation_steps and progress_log_every must be positive')
    if opt.debug and accumulation > 1:
        raise ValueError('Use smoke data with debug=False when testing gradient accumulation')
    
    current_time = time.localtime()
    hour = current_time.tm_hour
    minute = current_time.tm_min
    second = current_time.tm_sec
    if not opt.random_seed and not getattr(opt, 'fixed_run_id', False):
        try:
            opt.id = opt.id+'_'+opt.exp_name #for batch training
        except:
            opt.id = "debug"+str(hour)+str(minute)+str(second) #for debug
    print('Rename opt.id as "{}".'.format(opt.id))
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, 'train.log')
    logger.info('RUNTIME %s',runtime)
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))

    if not opt.start_from:
        backup_envir(save_folder, vars(opt))
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    model_pth = None
    resume_pending_validation = False
    if opt.start_from:
        opt.pretrain = False
        filename = 'model-best.pth' if opt.start_from_mode == 'best' else 'model-last.pth'
        # Load only trusted experiment checkpoints; format 2 contains Python/NumPy RNG state.
        model_pth = torch.load(os.path.join(save_folder, filename), map_location='cpu', weights_only=False)
        if model_pth.get('format_version') != 2:
            raise ValueError('Legacy checkpoint lacks a consistent scheduler/RNG boundary. Use a new run or an explicit migration; do not silently resume it.')
        validate_resume(vars(opt), model_pth['config'])
        saved_info = model_pth['info']
        resume_pending_validation = model_pth['phase'] == 'validation_pending'
        logger.info('Restore checkpoint phase %s, next epoch %s', model_pth['phase'], model_pth['next_epoch'])

    train_dataset = PropSeqDataset(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   opt.dict_file, True, 'gt',
                                   opt)

    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, 'gt',
                                 opt)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn,
                              pin_memory=getattr(opt,'pin_memory',False),
                              prefetch_factor=2 if opt.nthreads>0 else None)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn,
                            pin_memory=getattr(opt,'pin_memory',False),
                            prefetch_factor=2 if opt.nthreads>0 else None)

    epoch = model_pth['next_epoch'] if model_pth else 0
    iteration = model_pth['iteration'] if model_pth else 0
    optimizer_steps = model_pth.get('optimizer_steps', iteration) if model_pth else 0
    best_val_score = saved_info.get('best', {}).get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    # Build model

    model, criterion,contrastive_criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.train()
    # exit()
    # Recover the parameters
    if model_pth is not None:
        model.load_state_dict(model_pth['model'], strict=True)

    # Load the pre-trained model
    if opt.pretrain and (not opt.start_from):
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        pretrain_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        if opt.pretrain == 'encoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in pretrain_pth['model'].items() if encoder_filter(k)}
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == 'decoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in pretrain_pth['model'].items() if not encoder_filter(k)}
            model.load_state_dict(decoder_pth, strict=True)
            pass
        elif opt.pretrain == 'full':
            model.load_state_dict(pretrain_pth['model'], strict=True)
        else:
            raise ValueError("wrong value of opt.pretrain")

    model.to(opt.device)

    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)
    cl_schedule_time, cl_schedule_val = opt.cl_schedule_time, opt.cl_schedule_val
    if model_pth is not None:
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.load_state_dict(model_pth['scheduler'])
    opt.current_lr = optimizer.param_groups[0]['lr']

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message('Start training !', logger)

    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()
    weight_dict = criterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    #####
    if opt.able_ret:
        memory_bank=load_clip_memory_bank(opt)
        clip_gt=load_clip_gt(opt)
        memory_bank['vid_sent_embeds']=torch.tensor(memory_bank['vid_sent_embeds']).to('cuda')
        memory_bank["clip_gt"]=clip_gt
    else:
        memory_bank=None        
    
    if model_pth is not None:
        restore_rng(model_pth['rng'])
    if epoch >= opt.epoch and not resume_pending_validation:
        logger.info('Requested training budget is already complete.')
        return saved_info

    # Epoch-level iteration
    while True:
        # scheduled sampling rate update
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                opt.scheduled_sampling_max_prob)
            model.caption_head.ss_prob = opt.ss_prob

        print('lr:{}'.format(float(opt.current_lr)))
        cl_weight = schedule_contrastive(weight_dict, epoch, cl_schedule_time, cl_schedule_val,
                                         corrected=getattr(opt, 'corrected_query_contrastive', False))
        criterion.matcher.cost_cl = 0 if cl_weight == 0 else opt.set_cost_cl
        logger.info('Epoch %s effective loss weights: %s', epoch, weight_dict)
        saved_this_epoch, new_best = False, False
        epoch_start = time.time()
        if opt.device=='cuda':torch.cuda.reset_peak_memory_stats()
        logger.info('EPOCH_START epoch=%d/%d videos=%d micro_batch=%d accumulation=%d effective_batch=%d resume_validation=%s',
                    epoch+1, opt.epoch, len(train_dataset), opt.batch_size, accumulation,
                    opt.batch_size*accumulation, resume_pending_validation)
        optimizer.zero_grad(set_to_none=True)

        if opt.save_mode:
            save_vid_features=[]
        

        # opt.debug=True
        # Batch-level iteration
        for batch_index, dt in enumerate(tqdm([] if resume_pending_validation else train_loader, disable=opt.disable_tqdm)):
            if opt.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            iteration += 1

            dt = {key: _.to(opt.device,non_blocking=getattr(opt,'pin_memory',False)) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device,non_blocking=getattr(opt,'pin_memory',False)) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]

            dt = collections.defaultdict(lambda: None, dt)

            if opt.save_mode:
                encoded_feature = model(dt, criterion, opt.transformer_input_type,save_mode=opt.save_mode)
                save_vid_features.append(encoded_feature.detach().cpu())
                continue
            else:
                with amp_context(opt):
                    output, loss = model(dt, criterion,contrastive_criterion, opt.transformer_input_type,memory_bank)
            
            final_loss = sum(loss[k].float() * weight_dict[k] for k in loss.keys() if k in weight_dict)
            if not torch.isfinite(final_loss):raise RuntimeError('Non-finite training loss; stopping before optimizer update')
            # Normalize each accumulation window by its actual video count, including the final partial window.
            window_start = (batch_index // accumulation) * accumulation
            window_length = min(accumulation, len(train_loader) - window_start)
            (final_loss / window_length).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if batch_index == 0 or (batch_index + 1) % log_every == 0 or batch_index + 1 == len(train_loader):
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_index + 1) * (len(train_loader) - batch_index - 1)
                logger.info('PROGRESS epoch=%d/%d micro_batch=%d/%d optimizer_steps=%d loss=%.5f lr=%.8g elapsed_s=%.1f eta_epoch_s=%.1f',
                            epoch+1,opt.epoch,batch_index+1,len(train_loader),optimizer_steps,
                            final_loss.item(),optimizer.param_groups[0]['lr'],elapsed,eta)

            for loss_k,loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0)+ loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()


            losses_log_every = max(1, int(len(train_loader) / 3))

            if opt.debug:
                losses_log_every = 6

            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()

                logger.info(
                    "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                        .format(opt.id, iteration, epoch, loss_sum,
                                (end - start) / losses_log_every, bad_video_num))

                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()

        if not resume_pending_validation:
            train_seconds=time.time()-epoch_start
            perf=dict(phase='train',epoch=epoch,train_seconds=train_seconds,
                      videos_per_second=len(train_dataset)/max(train_seconds,1e-6),
                      optimizer_steps=optimizer_steps,precision=getattr(opt,'precision','fp32'),**vram_metrics(opt.device))
            append_performance(save_folder,perf)
            logger.info('TRAIN_PERFORMANCE %s',perf)
        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):

            # Save model
            saved_pth = snapshot(model, optimizer, lr_scheduler, epoch, iteration,
                                 saved_info, vars(opt), 'validation_pending', optimizer_steps=optimizer_steps)
            checkpoint_path = os.path.join(save_folder, 'model-last.pth')
            atomic_save(saved_pth, checkpoint_path)
            saved_this_epoch = True

            model.eval()
            logger.info('VALIDATION_START epoch=%d/%d videos=%d', epoch+1, opt.epoch, len(val_dataset))
            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}.json'.format(
                                             len(val_dataset), epoch))
            if epoch in [5,6,7,8,9,10,11,12,13,14,15,17,19,21,24,27]:
                infer_mode=True
            else:
                infer_mode=False
            infer_mode=True
            eval_score, eval_loss = evaluate(model,memory_bank, criterion,contrastive_criterion, postprocessors, val_loader, result_json_path, logger=logger, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug,infer_mode=infer_mode)
            if opt.caption_decoder_type == 'none':
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if opt.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

            for loss_type in eval_loss.keys():
                tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            # save the model parameter and  of best epoch
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': plain_metadata(vars(opt)),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': float(best_val_score),
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                new_best = True
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))
                logger.info('Save Best-model at epoch {} to checkpoint file.'.format(epoch))

            saved_info['last'] = {'opt': plain_metadata(vars(opt)),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': float(best_val_score),
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     # 'query_matched_fre_hist': query_matched_fre_hist,
                                     }
            atomic_save(saved_info, os.path.join(save_folder, 'info.json'), json_file=True)
            logger.info('Save info to info.json')

            model.train()

        completed_epoch = epoch
        epoch += 1
        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        resume_pending_validation = False
        if saved_this_epoch:
            # One self-contained state: model, optimizer, next-epoch scheduler, RNG and metadata.
            complete = snapshot(model, optimizer, lr_scheduler, completed_epoch, iteration,
                                saved_info, vars(opt), 'epoch_complete', optimizer_steps=optimizer_steps)
            atomic_save(complete, checkpoint_path)
            if new_best:
                atomic_save(complete, os.path.join(save_folder, 'model-best.pth'))
            if opt.save_all_checkpoint:
                atomic_save(complete, os.path.join(save_folder, 'model_epoch_{}.pth'.format(completed_epoch)))
            logger.info('EPOCH_COMPLETE epoch=%d/%d optimizer_steps=%d checkpoint=%s',
                        completed_epoch+1,opt.epoch,optimizer_steps,checkpoint_path)
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            tf_writer.close()
            break

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    if not opt.text_crossAttn:
        opt.cap_num_feature_levels=4
    if opt.combined_encoder:
        opt.cap_num_feature_levels=5
        
    if opt.feature_dim==3072:
        pass
    elif opt.feature_dim==512:
        if opt.decoder_query_ablation:
            opt.num_queries=opt.decoder_query_num
    
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False
    opt.rec_encdec=True
    
    if opt.caption_decoder_new=="gpt2":
        opt.caption_decoder_type="gpt2"
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    train(opt)
