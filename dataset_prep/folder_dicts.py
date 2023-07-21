import os
import json
from collections import defaultdict, OrderedDict


d=defaultdict(set)
for path,dirs,files in os.walk('/home/datasets/Hyper_Kvasir/100Frames200VidDataSemiPaired/trainA/'):
   d[os.path.basename(path)]=len(files)

d.pop("")
with open('train.json', 'w') as fp:
    json.dump(OrderedDict(sorted(d.items())), fp, indent=0)


d=defaultdict(set)
for path,dirs,files in os.walk('/home/datasets/Hyper_Kvasir/100Frames200VidDataSemiPaired/testA/'):
   d[os.path.basename(path)]=len(files)

d.pop("")
with open('test.json', 'w') as fp:
    json.dump(OrderedDict(sorted(d.items())), fp, indent=0)