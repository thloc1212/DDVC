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

class GPT2CaptionModel(nn.Module):
    def __init__(self, opt):
        super(GPT2CaptionModel, self).__init__()
        self.opt = opt
        if getattr(opt, 'local_visual_attention', False) and not opt.add_cross_attention:
            raise ValueError('local_visual_attention requires add_cross_attention=True')
        # prefix mlp
        self.fc1 = nn.Linear(self.opt.hidden_dim, self.opt.hidden_dim*2)
        self.fc2 = nn.Linear(self.opt.hidden_dim*2, 768)
        self.relu = nn.ReLU()
        self.tokenizer = GPT2Tokenizer.from_pretrained('/DATA_DISK/plm_cache/gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        configuration = GPT2Config(n_layer=self.opt.gpt_layer,add_cross_attention=self.opt.add_cross_attention)

        self.model = GPT2LMHeadModel(configuration)

        self.prefix_token=self.tokenizer("<|endoftext|>",return_tensors="pt")["input_ids"].to(opt.device)


    def input_project(self,hs):
        prefix_emb = self.relu(self.fc1(hs))
        prefix_emb = self.fc2(prefix_emb)
        return prefix_emb


    def visual_context(self, hs, clip_frame_feature, event_boxes=None, video_mask=None):
        if clip_frame_feature.shape[0] != 1:
            raise ValueError('DDVC caption head currently requires batch_size=1')
        # The vendored GPT-2 Conv1D uses view(), so it needs contiguous input.
        cross_emb = clip_frame_feature.repeat(hs.shape[0], 1, 1).detach()
        if video_mask is None:
            video_mask = torch.ones(1, cross_emb.shape[1], device=cross_emb.device, dtype=torch.bool)
        if getattr(self.opt, 'local_visual_attention', False):
            if event_boxes is None or len(event_boxes) != hs.shape[0]:
                raise ValueError('Each caption query needs its own predicted interval')
            mask = event_visual_mask(event_boxes, video_mask,
                getattr(self.opt, 'local_visual_context_ratio', 0.1),
                getattr(self.opt, 'local_visual_min_frames', 4))
        else:
            mask = video_mask.to(device=cross_emb.device, dtype=torch.bool).expand(hs.shape[0], -1)
        return cross_emb, mask

    def forward(self, hs, cap_label, cap_mask ,others, cap_tensor, clip_fram_feature,get_loss=True,
                event_boxes=None, video_mask=None):
        prefix_emb = self.relu(self.fc1(hs))
        prefix_emb = self.fc2(prefix_emb)
        input_emb=self.model.transformer.wte(cap_label)
        input_emb[:,:1,:]=prefix_emb
        if self.opt.add_cross_attention==False:
            if get_loss:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask, labels=cap_label)
            else:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask)
        else:
            cross_emb, cross_mask = self.visual_context(hs, clip_fram_feature, event_boxes, video_mask)
            if get_loss:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,labels=cap_label, encoder_hidden_states=cross_emb, encoder_attention_mask=cross_mask)
            else:
              output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,encoder_hidden_states=cross_emb, encoder_attention_mask=cross_mask)
        return output

    def generate(self,hs, reference, others, clip_fram_feature, event_boxes=None, video_mask=None):
        with torch.no_grad():
            prefix_emb = self.relu(self.fc1(hs))
            prefix_emb = self.fc2(prefix_emb)
            seqNum_prefix_token=self.prefix_token.repeat(hs.shape[0], 1)
            input_emb=self.model.transformer.wte(seqNum_prefix_token)
            input_emb[:,:1,:]=prefix_emb
            attention_mask=torch.ones(seqNum_prefix_token.shape,dtype=torch.long,device=input_emb.device)
            if self.opt.add_cross_attention==False:
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            else:
                cross_emb, cross_mask = self.visual_context(hs, clip_fram_feature, event_boxes, video_mask)
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,encoder_hidden_states=cross_emb,
                                                encoder_attention_mask=cross_mask,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            return output
