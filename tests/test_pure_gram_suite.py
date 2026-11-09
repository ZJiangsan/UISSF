from argparse import Namespace
from pathlib import Path
import json
from reproduction.build_experiments import generate


def test_pure_gram_jobs_zero_both_auxiliary_terms(tmp_path):
    args=Namespace(project=str(tmp_path/'project'),output_root=str(tmp_path/'results'),dataset='chikusei',phase='loss',epochs=100000,seeds=3,eval_every=1000,save_every=1000,log_every=100,objective='original',walltime='24:00:00',python='/trinity/home/jizh/venvs/nerfstudio/bin/python',check_inputs=False,submit=False,pure_gram_loss=True)
    jobs=generate(args)
    assert len(jobs)==7
    assert len({j['output'] for j in jobs})==7
    for j in jobs:
        script=Path(j['script']).read_text()
        assert '--reconstruction-weight 0 --entropy-weight 0' in script
        assert '--loss-variant workingone' in script
        assert '--reset-adam-each-epoch' in script
        assert '--epochs 100000' in script
        assert '#!/bin/bash' == script.splitlines()[0]
    meta=json.loads((tmp_path/'results/chikusei/loss/manifest.json').read_text())
    assert meta['pure_gram_loss'] is True
    assert meta['auxiliary_weights']=={'reconstruction':0.0,'entropy':0.0}
