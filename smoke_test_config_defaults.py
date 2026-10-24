"""Historical prepared configs must get the same defaults as training."""
import json
from pathlib import Path
import sys
import tempfile
import yaml
from opts import parse_opts


def main():
    config=json.loads((Path(__file__).parent/'opts.json').read_text())
    config.update(random_seed=False,debug=False,precision='bf16',
                  caption_cost_chunk_size=64,gradient_accumulation_steps=8)
    config.pop('contrastive_loss_coef',None)
    argv=sys.argv[:]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'config.yml'
        path.write_text(yaml.safe_dump(config))
        opt=parse_opts(['--cfg_path',str(path)],write_options=False)
        assert opt.contrastive_loss_coef==0.5
        assert (opt.precision,opt.caption_cost_chunk_size,opt.gradient_accumulation_steps)==('bf16',64,8)
        config['contrastive_loss_coef']=0.7
        path.write_text(yaml.safe_dump(config))
        assert parse_opts(['--cfg_path',str(path)],write_options=False).contrastive_loss_coef==0.7
    assert sys.argv==argv
    print('PASS training defaults + historical YAML overrides without consuming CLI argv')


if __name__=='__main__':main()
