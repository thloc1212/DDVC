"""Import verified shared data only; never import trained model weights."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import yaml

def import_c1_data(drive,helpers,dataset):
    """Reuse immutable old C1 assets, relocating config paths after byte verification.

    Never upload another CLIP bundle or touch the original cache pointer/weights.
    """
    parent=drive.path('data/'+dataset+'/c1-base-assets-v1')
    pointer=drive.optional_json(parent+'/LATEST.json')
    if pointer is None:return False
    generation=pointer.get('generation','')
    if not re.fullmatch('[0-9a-f]{32}',generation):raise ValueError('Invalid C1 data generation')
    remote=parent+'/generations/'+generation;meta=drive.read_json(remote+'/READY.json')
    if meta.get('protocol')!='c1-base-assets-v1' or meta.get('dataset')!=dataset or meta.get('asset_code')!=helpers.data_identity():
        raise ValueError('Original C1 data recipe does not match baseline experiment')
    old_work=PurePosixPath(meta['work']);old_base=old_work/'prepared'/(dataset+'_c1_base_assets')
    if not old_work.is_absolute() or '..' in old_work.parts:raise ValueError('Invalid original C1 root')
    destination=helpers.base_path(dataset)
    if destination.exists():raise FileExistsError('Partial local C1 data; preserve and inspect '+str(destination))
    with tempfile.TemporaryDirectory(dir=helpers.WORK,prefix='.baseline-data-') as temp:
        staged=Path(temp)/'base'
        helpers.fetch_verified(drive,remote,meta,staged,
            lambda n:Path(n).suffix in {'.json','.yml','.yaml','.npy','.pth'})
        original=yaml.safe_load((staged/'config.yml').read_text())
        if original.get('dataset')!=dataset or original.get('bank_type')!=[dataset]:raise ValueError('Dataset mismatch')
        for k in ('train_caption_file','val_caption_file','dict_file','bank_path'):
            if not PurePosixPath(original[k]).is_relative_to(old_base):raise ValueError('Data path escapes C1 base bundle')
        if len(original['visual_feature_folder'])!=1 or not PurePosixPath(original['visual_feature_folder'][0]).is_relative_to(old_base):
            raise ValueError('Feature path escapes C1 base bundle')
        actual=hashlib.sha256(b''.join((staged/PurePosixPath(original[k]).relative_to(old_base)).read_bytes()
                 for k in ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
        if actual!=original['data_signature']:raise ValueError('C1 annotation signature differs')
        def relocate(x):
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,str) and (x==str(old_work) or x.startswith(str(old_work)+'/')):
                return str(helpers.WORK)+x[len(str(old_work)):]
            return x
        (staged/'config.yml').write_text(yaml.safe_dump(relocate(original),sort_keys=False),encoding='utf8')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(staged),destination)
    helpers.write_json(helpers.WORK/'provenance'/('imported_data_'+dataset+'.json'),
                       dict(source=remote,original_manifest=meta,weights_imported=False))
    print('BASELINE_REUSED_VERIFIED_C1_ASSETS_NO_WEIGHTS',dataset,destination,flush=True)
    return True
