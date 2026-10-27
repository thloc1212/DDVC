"""Event concept supervision and predicted-concept conditioning for random GPT-2.

Inspired by concept cues in MCCL (AAAI 2025), not a reproduction of its frame
detector/cyclic learning. GT words are used only in objective(), after matching.
"""
import torch
from torch import nn
from torch.nn import functional as F
from misc.event_concepts import concept_words, load_vocabulary


class EventConceptConditioner(nn.Module):
    def __init__(self, opt):
        super().__init__()
        vocab = load_vocabulary(opt.concept_vocab_path, opt.concept_vocab_sha256)
        self.concepts = vocab['concepts']
        self.word_to_id = {word: index for index, word in enumerate(self.concepts)}
        self.condition_prefix = bool(opt.concept_condition_prefix)
        count = len(self.concepts)
        hidden, dim = int(opt.concept_hidden_dim), int(opt.concept_embedding_dim)
        if hidden < 1 or dim < 1:
            raise ValueError('Concept hidden/embedding dimensions must be positive')
        self.classifier = nn.Sequential(nn.Linear(opt.hidden_dim + opt.feature_dim, hidden),
                                        nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, count))
        frequency = torch.tensor(vocab['document_counts'], dtype=torch.float32)
        prior = (frequency / vocab['caption_count']).clamp(.001, .999)
        with torch.no_grad():
            self.classifier[-1].bias.copy_(torch.logit(prior))
        self.register_buffer('pos_weight', ((vocab['caption_count']-frequency)/frequency).clamp(1.,20.))
        self.embedding = nn.Parameter(torch.empty(count, dim))
        nn.init.normal_(self.embedding, std=.02)
        self.to_prefix = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(self.to_prefix.weight)  # Exact zero contribution at initialization.

    @staticmethod
    def pool_visual(features, boxes, video_mask):
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError('Event concepts follow the current DDVC batch_size=1 recipe')
        if boxes is None or boxes.ndim != 2 or boxes.shape[-1] != 2 or not len(boxes):
            raise ValueError('Need one predicted center/duration interval per event query')
        if video_mask is None:
            valid = torch.ones(features.shape[1], dtype=torch.bool, device=features.device)
        else:
            if video_mask.shape != features.shape[:2]: raise ValueError('Video mask shape mismatch')
            valid = video_mask[0].bool()
        length = int(valid.sum())
        if not length: raise ValueError('Cannot pool an empty video')
        boxes = boxes.detach().float()
        if not torch.isfinite(boxes).all(): raise ValueError('Non-finite concept intervals')
        positions = (torch.arange(length, device=boxes.device).float()+.5)/length
        start, end = boxes[:,0]-boxes[:,1]/2, boxes[:,0]+boxes[:,1]/2
        keep = (positions[None] >= start[:,None]) & (positions[None] <= end[:,None])
        nearest = (positions[None]-boxes[:,:1]).abs().argmin(-1, keepdim=True)
        fallback = torch.zeros_like(keep).scatter(1,nearest,True)
        keep = keep | (fallback & ~keep.any(-1,keepdim=True))
        weights = keep.float()/keep.sum(-1,keepdim=True).clamp_min(1)
        # Pre-extracted CLIP stays frozen; hard predicted pooling does not regress boxes.
        return weights @ features[0,valid].detach().float()

    def forward(self, hs, features, boxes, video_mask):
        if hs.ndim != 3 or hs.shape[1] != 1 or hs.shape[0] != len(boxes):
            raise ValueError('Expected caption queries [events,1,hidden_dim]')
        pooled = self.pool_visual(features, boxes, video_mask)
        logits = self.classifier(torch.cat((hs[:,0],pooled.to(hs.dtype)),dim=-1))
        probabilities = logits.float().sigmoid()
        weights = probabilities/probabilities.sum(-1,keepdim=True).clamp_min(1e-6)
        semantic = weights.to(self.embedding.dtype) @ self.embedding
        delta = self.to_prefix(semantic)
        if not self.condition_prefix: delta = delta*0.
        return logits, delta

    def objective(self, hs, features, boxes, video_mask, captions):
        logits, delta = self(hs,features,boxes,video_mask)
        if len(captions) != len(logits): raise ValueError('Matched concept/caption count differs')
        target = torch.zeros_like(logits, dtype=torch.float32)
        for row, sentence in enumerate(captions):
            ids = [self.word_to_id[word] for word in concept_words(sentence) if word in self.word_to_id]
            if ids: target[row,ids] = 1.
        supported = target.sum(-1) > 0
        # Unsupported captions are not treated as all-negative semantic examples.
        loss = (F.binary_cross_entropy_with_logits(logits[supported].float(),target[supported],
                  pos_weight=self.pos_weight.float()) if supported.any() else logits.sum()*0.)
        with torch.no_grad():
            top_ids=logits.topk(min(10,len(self.concepts)),dim=-1).indices
            recalled=target.gather(1,top_ids).sum(-1)/target.sum(-1).clamp_min(1)
            recall=recalled[supported].mean() if supported.any() else logits.new_tensor(0.)
        return dict(loss_event_concept=loss,
                    concept_target_coverage=supported.float().mean().detach(),
                    concept_mean_probability=logits.float().sigmoid().mean().detach(),
                    concept_prefix_norm=delta.float().norm(dim=-1).mean().detach(),
                    concept_top10_recall=recall)
