import json
import glob
import numpy as np

templates = []
templates.append('baseline_*')
templates.append('luketina_*')
templates.append('IFT_3_*')
templates.append('IFT_2_*')
templates.append('proposed_0.9_*')
templates.append('proposed_0.99_*')
templates.append('proposed_0.5_*')


for t in templates:
    acc = [list(json.loads(open(f).read()).values())[0] for f in glob.glob(t)]
    if len(acc) == 0:
        continue
    print(t, '\t', round(np.mean(acc) * 100, 2), '\t+-', round(np.std(acc) / np.sqrt(len(acc)) * 100, 2))
