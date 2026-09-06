#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.
"""Alternate Metal ablations on benchmark.py --prepare inputs; keep every sample."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from benchmark import compare

VARIANTS = {
    'current': {},
    'legacy_copy': {'FCPE_METAL_LEGACY_COPY':'1'},
    'legacy_graph': {'FCPE_METAL_LEGACY_GRAPH':'1'},
    'legacy_dw_layout': {'FCPE_METAL_LEGACY_DW_LAYOUT':'1'},
    'legacy_dw_kernel': {'FCPE_METAL_LEGACY_DW':'1','FCPE_METAL_LEGACY_DW_LAYOUT':'1'},
    'legacy_gn': {'FCPE_METAL_LEGACY_GN':'1'},
    'legacy_im2col': {'FCPE_METAL_LEGACY_IM2COL':'1'},
    'no_glu_project': {'FCPE_METAL_DISABLE_GLU_PROJECT':'1'},
    'no_fusion': {'FCPE_METAL_DISABLE_FUSION':'1'},
    'single_submit': {'FCPE_METAL_SINGLE_SUBMIT':'1'},
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench',type=Path,default=Path('build-metal/bin/fcpe-bench'))
    p.add_argument('--baseline-bench',type=Path)
    p.add_argument('--model',type=Path,default=Path('models/fcpe-f32.gguf'))
    p.add_argument('--inputs',type=Path,default=Path('validation/macos/benchmark'))
    p.add_argument('--output-dir',type=Path,default=Path('validation/metal-ablation'))
    p.add_argument('--backend',default='MTL0')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    a=p.parse_args()
    if a.rounds<1 or a.repeats<1 or a.warmup<0 or a.threads<1:p.error('invalid measurement settings')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.inputs/'inputs.json').read_text())
    cases=[c for c in manifest['cases'] if not a.cases or c['name'] in a.cases]
    if not cases or (a.cases and set(a.cases)!={c['name'] for c in cases}):p.error('unknown cases')
    names=list(a.variants)+(['baseline'] if a.baseline_bench else [])
    reports=[]
    for round in range(a.rounds):
        order=names if round%2==0 else names[::-1]
        for case in cases:
            ref=np.fromfile(a.inputs/f"{case['name']}.reference.f32",dtype='<f4').reshape(-1,360)
            cents=np.fromfile(a.inputs/'cents.f32',dtype='<f4')
            mel=a.inputs/f"{case['name']}.mel.f32"
            if sha(mel)!=case['mel_sha256']:raise ValueError('input hash mismatch')
            for name in order:
                bench=(a.baseline_bench if name=='baseline' else a.bench).resolve()
                prefix=a.output_dir/f"round{round}-{name}-{case['name']}"
                env=os.environ.copy()
                for k in list(env):
                    if k.startswith(('FCPE_METAL_','GGML_METAL_')):env.pop(k)
                flags=VARIANTS.get(name,{})
                env.update(flags)
                if name=='baseline':
                    # CMake build executables contain an absolute build-tree rpath.
                    # Pin all dependencies to the copied baseline directory.
                    env['DYLD_LIBRARY_PATH']=str(bench.parent)
                    env['DYLD_PRINT_LIBRARIES']='1'
                cmd=[str(bench),'--model',str(a.model.resolve()),'--mel',str(mel.resolve()),'--backend',a.backend,
                     '--threads',str(a.threads),'--warmup',str(a.warmup),'--repeats',str(a.repeats),
                     '--output',str(prefix)+'.json','--dump',str(prefix)+'.f32',
                     '--wav',str((a.inputs/f"{case['name']}.wav").resolve())]
                result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
                Path(str(prefix)+'.log').write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(result.stderr[-6000:])
                if name=='baseline':
                    loads=[l for l in result.stderr.splitlines() if l.startswith('dyld[') and 'libggml' in l]
                    if len(loads)<4 or any(str(bench.parent) not in l for l in loads):raise RuntimeError('baseline dylib isolation failed')
                r=json.loads(Path(str(prefix)+'.json').read_text())
                got=np.fromfile(str(prefix)+'.f32',dtype='<f4').reshape(-1,360)
                accuracy=compare(got,ref,cents)
                record={'round':round,'variant':name,'environment':flags,'input':case,**r,'accuracy':accuracy,
                        'executable_sha256':sha(bench),
                        'library_sha256':{f.name:sha(f) for f in sorted(bench.parent.glob('*.0.19.0.dylib'))}}
                reports.append(record)
                (a.output_dir/'ablation.json').write_text(json.dumps({'rounds':a.rounds,'warmup':a.warmup,'repeats':a.repeats,
                    'model_sha256':sha(a.model),'inputs':manifest,'results':reports},indent=2)+'\n')
                if accuracy['probability_max_abs']>1e-5 or accuracy['uv_mismatch'] or accuracy['f0_max_abs_hz']>0.02:
                    raise RuntimeError(f'{name}/{case["name"]}: numerical failure; reports retained')
                print(round,name,case['name'],f"{r['median_ms']:.3f} ms",flush=True)

if __name__=='__main__':main()
