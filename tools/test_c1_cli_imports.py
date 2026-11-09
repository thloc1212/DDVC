"""Run actual CLI entry points with Python safe-path and an unrelated cwd."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parent


class SafePathTests(unittest.TestCase):
    def test_entrypoints_bootstrap_sibling_imports(self):
        env=dict(os.environ,PYTHONSAFEPATH='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
        # Preserve dependency paths, but do not rescue the tool imports via PYTHONPATH.
        paths=env.get('PYTHONPATH','').split(os.pathsep)
        env['PYTHONPATH']=os.pathsep.join(p for p in paths if p and Path(p).resolve()!=TOOLS)
        with tempfile.TemporaryDirectory() as cwd:
            for name in ('c1_diagnostics.py','c1_visual_probe.py','c1_pipeline.py'):
                with self.subTest(name=name):
                    result=subprocess.run([sys.executable,'-P','-B',str(TOOLS/name),'--help'],
                                          cwd=cwd,env=env,text=True,capture_output=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                    self.assertIn('usage:',result.stdout)


if __name__=='__main__':unittest.main()
