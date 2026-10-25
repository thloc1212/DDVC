from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import torch
import numpy as np
import json
from collections import OrderedDict
from tqdm import tqdm
from os.path import dirname, abspath
import math
from misc.proposal_selection import select_proposals
from misc.runtime_perf import amp_context

ddvc_dir = dirname(abspath(__file__))
sys.path.insert(0, ddvc_dir)
sys.path.insert(0, os.path.join(ddvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(ddvc_dir, 'densevid_eval3/SODA'))


from densevid_eval3.eval_soda import eval_soda
from densevid_eval3.eval_para import eval_para
from densevid_eval3.eval_dvc import eval_dvc
from densevid_eval3.eval_tal import eval_tal

from misc.plot_proposal_distribution import main as plot_proposal_distribution
from densevid_eval3.eval_grounding import eval_result as eval_grounding

def calculate_avg_proposal_num(json_path):
    data = json.load(open(json_path))
    return np.array([len(v) for v in data['results'].values()]).mean()

def convert_tapjson_to_dvcjson(tap_json, dvc_json):
    data = json.load(open(tap_json, 'r'))
    data['version'] = "VERSION 1.0"
    data['external_data'] = {'used:': True, 'details': "C3D pretrained on Sports-1M"}

    all_names = list(data['results'].keys())
    for video_name in all_names:
        for p_info in data['results'][video_name]:
            p_info['timestamp'] = p_info.pop('segment')
            p_info['proposal_score'] = p_info.pop('score')
            p_info['sentence_score'] = p_info.pop('sentence_score', 0)
        data['results']["v_" + video_name] = data['results'].pop(video_name)
    json.dump(data, open(dvc_json, 'w'))


def convert_dvcjson_to_tapjson(dvc_json, tap_json):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        event_num = len(data[video_name])
        timestamps = [data[video_name][i]['timestamp'] for i in range(event_num)]
        sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            score = data[video_name][i].get('proposal_score', 1.0)
            video_info.append({'segment': timestamp, 'score': score, 'sentence': sentences[i], 'sentence_score': data[video_name][i].get('sentence_score', 0)})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))


def convert_gtjson_to_tapjson(gt_json, tap_json):
    data = json.load(open(gt_json, 'r'))
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        timestamps = data[video_name]['timestamps']
        sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1., 'sentence': sentences[i]})
        out['results'][video_name[2:]] = video_info
    with open(tap_json, 'w') as f:
        json.dump(out, f)


def get_topn_from_dvcjson(dvc_json, out_json, top_n=3, ranking_key='proposal_score', score_thres=-1e8):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}
    all_names = list(data.keys())
    num = 0
    bad_vid = 0
    for video_name in all_names:
        info = data[video_name]
        new_info = sorted(info, key=lambda x: x[ranking_key], reverse=True)
        new_info = [p for p in new_info if p[ranking_key] > score_thres]
        new_info = new_info[:top_n]
        out['results'][video_name] = new_info
        num += len(new_info)
        if len(new_info) == 0:
            bad_vid += 1
            out['results'].pop(video_name)
    print('average proosal number: {}'.format(num / len(all_names)))
    print('bad videos number: {}'.format(bad_vid))
    print('good videos number: {}'.format(len(out['results'])))
    with open(out_json, 'w') as f:
        json.dump(out, f)

def eval_metrics_grounding(g_filename, gt_filename):
    score = collections.defaultdict(lambda: -1)
    grounding_scores = eval_grounding(g_filename, gt_filename)
    for key in grounding_scores.keys():
        score['grounding_'+key] = float(grounding_scores[key])
    return score

def eval_metrics(dvc_filename, gt_filenames, para_gt_filenames, alpha=0.3, temperature=2.0, cl_score_weight=0., ranking_key='proposal_score', rerank=False, dvc_eval_version='2018',infer_mode=False):
    score = collections.defaultdict(lambda: -1)
    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames, version=dvc_eval_version,infer_mode=infer_mode)
    dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    dvc_score.update({'MetaScore': dvc_score['METEOR'] + dvc_score['soda_c']})
    score.update(dvc_score)
    return score


def save_dvc_json(out_json, path):
    with open(path, 'w') as f:
        out_json['valid_video_num'] = len(out_json['results'])
        out_json['avg_proposal_num'] = np.array([len(v) for v in out_json['results'].values()]).mean().item()
        json.dump(out_json, f)

def reranking(p_src, alpha, cl_score_weight, temperature, fix_topN=-1, increase_num=0,
              method='topk', sigma=.5, merge=False, merge_iou=.7, merge_text=.9):
    print('SELECTION method={} sigma={} merge={} alpha={} temp={}'.format(method,sigma,merge,alpha,temperature))
    with open(p_src) as stream:d=json.load(stream)
    for key,rows in d['results'].items():
        if not rows:continue
        top_n=rows[0]['pred_event_count'] if fix_topN<0 else fix_topN
        fraction=increase_num-math.floor(increase_num)
        extra=math.floor(increase_num)+(np.random.binomial(1,p=fraction,size=None) if fraction>0 else 0)
        d['results'][key]=select_proposals(rows,int(top_n+extra),method,sigma,alpha,temperature,
                                          merge,merge_iou,merge_text)
    d['selection']={'method':method,'sigma':sigma,'merge':merge,'merge_iou':merge_iou,'merge_text_similarity':merge_text}
    suffix='' if method=='topk' and not merge else '_{}_sigma{}_merge{}'.format(method,sigma,int(merge))
    save_path=p_src+'_rerank_alpha{}_temp{}{}.json'.format(alpha,temperature,suffix)
    save_dvc_json(d,save_path)
    return save_path


def evaluate(model,memory_bank,criterion,contrastive_criterion, postprocessors, loader, dvc_json_path, logger=None, score_threshold=0,
             alpha=0.3, dvc_eval_version='2018', device='cuda', debug=False, skip_lang_eval=False,infer_mode=False,sent_embedder=None, tokenizer=None):
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': None}}
    
    out_json_g = {'results': {}}
    aux_out_json_g = {'results': {}}

    opt = loader.dataset.opt
    
    loss_sum = OrderedDict()
    len_check=0
    with torch.set_grad_enabled(False):
        for dt in tqdm(loader, disable=opt.disable_tqdm):
            len_check+=1
            dt = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}

            dt['video_target'] = [
                    {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                    dt['video_target']]
            gt_bank=None
            if opt.ideal_test:
                gt_bank=load_val_gt(opt)
                gt_bank['vid_sent_embeds'] = torch.tensor(gt_bank['vid_sent_embeds']).to('cuda')
            
            captions = list()
            for video_sents in dt['cap_raw']:
                captions.extend(video_sents)


            with amp_context(opt):
                output, loss = model(dt, criterion, contrastive_criterion, opt.transformer_input_type,memory_bank, eval_mode=True,sent_embedder=sent_embedder,gt_bank=gt_bank)
            if logger is not None and (len_check == 1 or len_check % max(1,getattr(opt,'progress_log_every',10)) == 0 or len_check == len(loader)):
                logger.info('VALIDATION_PROGRESS batches=%d/%d',len_check,len(loader))
            orig_target_sizes = dt['video_length'][:, 1]

            weight_dict = criterion.weight_dict
            final_loss = sum(loss[k].float() * weight_dict[k] for k in loss.keys() if k in weight_dict)

            for loss_k, loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0) + loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            results = postprocessors['bbox'](output, orig_target_sizes, loader, model, tokenizer)
            
            results_g, cl_scores = postprocessors['bbox'].forward_grounding(output, orig_target_sizes, dt['video_target'])
            aux_results_g, aux_cl_scores = postprocessors['bbox'].forward_grounding(output['aux_outputs'][-1], orig_target_sizes, dt['video_target'])
            batch_json = {}
            batch_json_g = {}
            aux_batch_json_g = {}
            for idx, video_name in enumerate(dt['video_key']):
                segment = results[idx]['boxes'].float().cpu().numpy()
                is_gt_proposals = opt.transformer_input_type == 'gt_proposals'
                segment_num = len(segment)
                raw_boxes = results[idx]['raw_boxes'].float().cpu().numpy()
                raw_boxes_mask = raw_boxes.sum(1) != 0
                
                batch_json[video_name] = [
                    {
                        "timestamp": segment[pid].tolist(),
                        "raw_box": raw_boxes[pid].tolist(),
                        "label": results[idx]['labels'][pid].item(),
                        
                        "proposal_score": results[idx]['scores'][pid].item(),
                        "sentence": results[idx]['captions'][pid],
                        "sentence_score": results[idx]['caption_scores'][pid],
                        "cl_score": results[idx]['cl_scores'][pid],
                        'query_id': results[idx]['query_id'][pid].item(),
                        'vid_duration': results[idx]['vid_duration'].item(),
                        'pred_event_count': results[idx]['pred_seq_len'].item(),
                    }
                    for pid in range(segment_num) if results[idx]['scores'][pid].item() > score_threshold and raw_boxes_mask[pid]]
                if results_g:
                    collect_grounding_result(idx, video_name, opt, dt, batch_json_g, results_g)
                if aux_results_g:
                    collect_grounding_result(idx, video_name, opt, dt, aux_batch_json_g, aux_results_g)
                
            out_json['results'].update(batch_json)
            out_json_g['results'].update(batch_json_g)
            aux_out_json_g['results'].update(aux_batch_json_g)
            if debug and len(out_json['results']) > 5:
                break

    if opt.only_ft_class_head:
        tal_result_json_path = dvc_json_path[:-5] + '.tal.json'
        out_json_tal = collect_tal_result(out_json, loader.dataset.name_map)
        save_dvc_json(out_json_tal, tal_result_json_path)  
    
    # Keep every decoder candidate for diagnosis, but preserve the final
    # metric input at dvc_json_path. Previously the raw proposals were saved
    # over the reranked/top-N file after evaluation, making the saved result
    # disagree with the reported metrics.
    raw_dvc_json_path = dvc_json_path[:-5] + '.raw.json' if dvc_json_path.endswith('.json') else dvc_json_path + '.raw.json'
    save_dvc_json(out_json, raw_dvc_json_path)
    metric_json_path = raw_dvc_json_path
    
    for k in loss_sum.keys():
        loss_sum[k] = np.round(loss_sum[k] / (len(loader) + 1e-5), 3).item()
    
    
    if logger is not None:
        logger.info('loss: {}'.format(loss_sum))

    if opt.count_loss_coef > 0:
        metric_json_path = reranking(raw_dvc_json_path, alpha=alpha, cl_score_weight=opt.eval_matching_score_weight, temperature=2.0,
            method=getattr(opt,'eval_selection_method','topk'),sigma=getattr(opt,'eval_softnms_sigma',.5),
            merge=getattr(opt,'eval_merge_duplicates',False),merge_iou=getattr(opt,'eval_merge_iou',.7),
            merge_text=getattr(opt,'eval_merge_text_similarity',.9))
    elif getattr(opt,'eval_selection_method','topk')!='topk' or getattr(opt,'eval_merge_duplicates',False):
        raise ValueError('Soft-NMS/merge requires positive count_loss_coef and predicted event counts')
    save_dvc_json(out_json_g, dvc_json_path + '.grounding.json')
    save_dvc_json(aux_out_json_g, dvc_json_path + '_aux.grounding.json')
    skip_lang_eval = skip_lang_eval or vars(opt).get('eval_disable_captioning', False)
    if not skip_lang_eval:
        scores = eval_metrics(metric_json_path,
                              gt_filenames=opt.gt_file_for_eval,
                              para_gt_filenames=opt.gt_file_for_para_eval,
                              alpha=alpha,
                              cl_score_weight=opt.eval_matching_score_weight,
                              rerank=(opt.count_loss_coef > 0),
                              dvc_eval_version=dvc_eval_version,
                              infer_mode=infer_mode
                              )
    else:
        scores = {}
    scores_g = eval_metrics_grounding(dvc_json_path + '.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)
    aux_scores_g = eval_metrics_grounding(dvc_json_path + '_aux.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)
    rename_aux_scores_g = {'aux_' + key: value for key, value in aux_scores_g.items()}
    out_json_g.update(scores_g)
    aux_out_json_g.update(aux_scores_g)
    scores.update(scores_g)
    scores.update(rename_aux_scores_g)
    if opt.only_ft_class_head:
        score_tal = eval_tal(ground_truth_filename=opt.tal_gt_file, prediction_filename=tal_result_json_path)
        out_json_tal.update(score_tal)
        save_dvc_json(out_json_tal, tal_result_json_path)
        scores.update(score_tal) 

    # Save the same selected proposals that were passed to the evaluator.
    final_json = json.load(open(metric_json_path, 'r'))
    final_json.update(scores)
    save_dvc_json(final_json, dvc_json_path)
    try:
        plot_proposal_distribution(dvc_json_path)
    except Exception as exc:
        if logger is not None:
            logger.warning('Could not plot proposal distribution: {}'.format(exc))
    return scores, loss_sum

def collect_tal_result(out, name_map):
    tal_out = {'results':{}, 'version':'VERSION 1.3', 'external_data':{}}
    for key, items in out['results'].items():
        key = key[2:]
        tal_items = []
        for pred in items:
            label = pred['label']
            segment = pred['timestamp']
            score = pred['proposal_score']
            tal_item = {
                'label':name_map.convert_idx2name(label),
                'segment':segment,
                'score':score
            }
            tal_items.append(tal_item)
        tal_out['results'].update({key: tal_items})
    return tal_out


def collect_grounding_result(idx, video_name, opt, dt, batch_json_g, results_g):
    for pid in range(len(results_g[idx]['boxes'])):
        v_name = video_name[2:] if len(video_name) > 11 else video_name
        batch_json_g[v_name + '-' + str(pid)] = [{
            "timestamp": results_g[idx]['boxes'][pid],
            "score": results_g[idx]['confs'][pid],
            "cl_score": results_g[idx]['cl_scores'][pid],
            "sentence": dt['cap_raw'][idx][pid]
        }]

def load_val_gt(args):
    first_iter=True
    for bank_type in args.bank_type: 
        rag = os.path.join(args.bank_path,bank_type) #which domain will be used
        rag = os.path.join(rag,args.visual_feature_type[0]) #matching for backbone

        if args.memory_type == 'clip':
            text_embed = np.load(os.path.join(rag,"gt_embeds.npy"))
            scene_videoID = np.load(os.path.join(rag,"gt_scene_videoID.npy"))
        if first_iter:
            scene_videoIDs = scene_videoID
            text_embeds = text_embed
        else:
            text_embeds = np.concatenate((text_embeds, text_embed))
            scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
        first_iter=False    
  
    gt_bank={}
    if len(text_embeds) == len(scene_videoIDs):
        gt_bank = {
                "vid_sent_embeds": text_embeds,
                "video_id": scene_videoIDs
            }
    else:
       
        return -1
    return gt_bank

