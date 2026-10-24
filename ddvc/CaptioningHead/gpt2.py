# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""



import torch
import torch.nn as nn
from transformers import  GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from .local_visual import event_visual_mask
from .proposal_mask import LearnedGatedProposalMask
from .temporal_caption import TemporalCaptionConditioning

class GPT2CaptionModel(nn.Module):
    def __init__(self, opt):
        super(GPT2CaptionModel, self).__init__()
        self.opt = opt
        if getattr(opt, 'local_visual_attention', False) and not opt.add_cross_attention:
            raise ValueError('local_visual_attention requires add_cross_attention=True')
        if getattr(opt, 'learned_proposal_mask', False) and (not opt.add_cross_attention or getattr(opt, 'local_visual_attention', False)):
            raise ValueError('learned_proposal_mask requires cross attention and local_visual_attention=False')
        # prefix mlp
        self.fc1 = nn.Linear(self.opt.hidden_dim, self.opt.hidden_dim*2)
        self.fc2 = nn.Linear(self.opt.hidden_dim*2, 768)
        self.relu = nn.ReLU()
        self.tokenizer = GPT2Tokenizer.from_pretrained('/DATA_DISK/plm_cache/gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        configuration = GPT2Config(n_layer=self.opt.gpt_layer,add_cross_attention=self.opt.add_cross_attention)

        self.model = GPT2LMHeadModel(configuration)
        self.proposal_mask = None
        self.temporal_conditioner = None
        self.temporal_diagnostics = {}
        time_prefix=getattr(opt,'caption_time_conditioning',False)
        frame_time=getattr(opt,'caption_frame_time_embedding',False)
        if time_prefix or frame_time:
            if (not opt.add_cross_attention or getattr(opt,'learned_proposal_mask',False)
                    or getattr(opt,'local_visual_attention',False)):
                raise ValueError('Temporal caption experiment requires global, ungated visual cross-attention')
            # Keep backbone RNG identical across global/time/frame ablations.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(1732)
                self.temporal_conditioner=TemporalCaptionConditioning(
                    bands=getattr(opt,'caption_time_bands',8),hidden=getattr(opt,'caption_time_hidden',128),
                    time_prefix=time_prefix,frame_positions=frame_time,
                    prefix_scale=getattr(opt,'caption_time_scale',1.),
                    frame_scale=getattr(opt,'caption_frame_time_scale',1.),
                    detach_boxes=getattr(opt,'caption_time_detach_boxes',True))
        if getattr(opt, 'learned_proposal_mask', False):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(1731)
                self.proposal_mask = LearnedGatedProposalMask(opt.proposal_mask_bins, opt.proposal_mask_encoding_dim)

        self.prefix_token=self.tokenizer("<|endoftext|>",return_tensors="pt")["input_ids"].to(opt.device)


    def input_project(self,hs):
        prefix_emb = self.relu(self.fc1(hs))
        prefix_emb = self.fc2(prefix_emb)
        return prefix_emb

    def caption_prefix(self,hs,event_boxes=None):
        # input_project stays semantic-only for the existing CLIP alignment loss.
        prefix=self.input_project(hs)
        conditioner=getattr(self,'temporal_conditioner',None)
        if conditioner is not None and conditioner.prefix_mlp is not None:
            if event_boxes is None or len(event_boxes)!=len(hs):
                raise ValueError('Each caption prefix needs its own predicted interval')
            delta=conditioner.prefix_delta(event_boxes).to(prefix.dtype)
            if not hasattr(self,'temporal_diagnostics'):self.temporal_diagnostics={}
            self.temporal_diagnostics.update({
                'diag_caption_time_prefix_norm':delta.detach().norm(dim=-1).mean(),
                'diag_caption_time_prefix_ratio':(delta.detach().norm(dim=-1)/prefix.detach().norm(dim=-1).clamp_min(1e-8)).mean()})
            prefix=prefix+delta
        return prefix


    def visual_context(self, hs, clip_frame_feature, event_boxes=None, video_mask=None, event_scores=None, anchor_boxes=None):
        if clip_frame_feature.shape[0] != 1:
            raise ValueError('DDVC caption head currently requires batch_size=1')
        # The vendored GPT-2 Conv1D uses view(), so it needs contiguous input.
        cross_emb = clip_frame_feature.repeat(hs.shape[0], 1, 1).detach()
        if video_mask is None:
            video_mask = torch.ones(1, cross_emb.shape[1], device=cross_emb.device, dtype=torch.bool)
        conditioner=getattr(self,'temporal_conditioner',None)
        if conditioner is not None and conditioner.frame_projection is not None:
            delta=conditioner.frame_delta(video_mask).to(cross_emb.dtype)
            valid=video_mask.bool()
            if not hasattr(self,'temporal_diagnostics'):self.temporal_diagnostics={}
            self.temporal_diagnostics.update({
                'diag_caption_frame_time_norm':delta.detach().norm(dim=-1)[valid].mean(),
                'diag_caption_frame_time_ratio':(delta.detach().norm(dim=-1)/clip_frame_feature.detach().norm(dim=-1).clamp_min(1e-8))[valid].mean()})
            cross_emb=cross_emb+delta
        if getattr(self.opt, 'learned_proposal_mask', False):
            if event_boxes is None or len(event_boxes) != hs.shape[0]:
                raise ValueError('Learned masks need one interval per caption query')
            details = self.proposal_mask(event_boxes, video_mask, event_scores, anchor_boxes)
            # Detach frozen CLIP features BEFORE gating, so caption gradients
            # still reach the learned mask, predicted interval and event score.
            cross_emb = (cross_emb * details['weights'].unsqueeze(-1)).contiguous()
            mask = video_mask.to(device=cross_emb.device, dtype=torch.bool).expand(hs.shape[0], -1)
        elif getattr(self.opt, 'local_visual_attention', False):
            if event_boxes is None or len(event_boxes) != hs.shape[0]:
                raise ValueError('Each caption query needs its own predicted interval')
            mask = event_visual_mask(event_boxes, video_mask,
                getattr(self.opt, 'local_visual_context_ratio', 0.1),
                getattr(self.opt, 'local_visual_min_frames', 4))
        else:
            mask = video_mask.to(device=cross_emb.device, dtype=torch.bool).expand(hs.shape[0], -1)
        return cross_emb, mask

    def forward(self, hs, cap_label, cap_mask ,others, cap_tensor, clip_fram_feature,get_loss=True,
                event_boxes=None, video_mask=None, event_scores=None, anchor_boxes=None):
        self.temporal_diagnostics={}
        prefix_emb = self.caption_prefix(hs,event_boxes)
        input_emb=self.model.transformer.wte(cap_label)
        input_emb[:,:1,:]=prefix_emb
        if self.opt.add_cross_attention==False:
            if get_loss:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask, labels=cap_label)
            else:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask)
        else:
            cross_emb, cross_mask = self.visual_context(hs, clip_fram_feature, event_boxes, video_mask, event_scores, anchor_boxes)
            if get_loss:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,labels=cap_label, encoder_hidden_states=cross_emb, encoder_attention_mask=cross_mask)
            else:
              output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,encoder_hidden_states=cross_emb, encoder_attention_mask=cross_mask)
        return output

    def generate(self,hs, reference, others, clip_fram_feature, event_boxes=None, video_mask=None, event_scores=None, anchor_boxes=None):
        with torch.no_grad():
            self.temporal_diagnostics={}
            prefix_emb = self.caption_prefix(hs,event_boxes)
            seqNum_prefix_token=self.prefix_token.repeat(hs.shape[0], 1)
            input_emb=self.model.transformer.wte(seqNum_prefix_token)
            input_emb[:,:1,:]=prefix_emb
            attention_mask=torch.ones(seqNum_prefix_token.shape,dtype=torch.long,device=input_emb.device)
            if self.opt.add_cross_attention==False:
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            else:
                cross_emb, cross_mask = self.visual_context(hs, clip_fram_feature, event_boxes, video_mask, event_scores, anchor_boxes)
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,encoder_hidden_states=cross_emb,
                                                encoder_attention_mask=cross_mask,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            return output
