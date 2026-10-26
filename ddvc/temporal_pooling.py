"""Pool half-open frame intervals without taking the mean of an empty slice."""
import torch


def pool_temporal_segments(features, spans):
    if features.ndim!=2 or features.shape[0]<1:
        raise ValueError('Expected a nonempty [frames, channels] feature tensor')
    length=features.shape[0]
    pooled=[]
    for span in spans:
        start,end=int(span[0]),int(span[1])
        # Preserve every valid, nonempty legacy slice. Sub-frame GT events can
        # collapse to [i, i] after quantization; use that nearest available frame.
        start=min(max(start,0),length-1)
        end=min(max(end,start+1),length)
        pooled.append(features[start:end].mean(dim=0))
    if not pooled:return features.new_empty((0,features.shape[1]))
    return torch.stack(pooled)
