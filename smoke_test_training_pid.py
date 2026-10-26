"""Exercise Linux process evidence using fixtures, without signalling processes."""
import importlib.util
from pathlib import Path
import tempfile


def main():
    spec=importlib.util.spec_from_file_location('pid_guard',Path(__file__).parent/'tools/training_pid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);proc=root/'proc';proc.mkdir();pidfile=root/'run.pid';config=root/'config.yml'
        assert module.active_training_pid(pidfile,config,proc) is None
        pidfile.write_text('123');assert module.active_training_pid(pidfile,config,proc) is None
        process=proc/'123';process.mkdir()
        argv=['/tmp/uv-venv/bin/python','-u','train.py','--cfg_path',str(config.resolve())]
        def fixture(state,args):
            (process/'stat').write_text('123 (python) '+state+' 1 2 3')
            (process/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args)+b'\0')
        fixture('Z',[]);assert module.active_training_pid(pidfile,config,proc) is None
        fixture('S',argv);assert module.active_training_pid(pidfile,config,proc)==123
        fixture('R',['python','other.py']);assert module.active_training_pid(pidfile,config,proc) is None
        fixture('R',argv[:-1]+['other_config.yml']);assert module.active_training_pid(pidfile,config,proc) is None
        fixture('S',['python','/repo/train.py','--cfg_path='+str(config.resolve())])
        assert module.active_training_pid(pidfile,config,proc)==123
        pidfile.write_text('not-a-pid');assert module.active_training_pid(pidfile,config,proc) is None
    print('PASS PID guard: zombies, exited/reused/malformed PIDs do not block; matching live training does block; no signals sent')


if __name__=='__main__':main()
