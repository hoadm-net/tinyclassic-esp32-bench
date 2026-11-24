import os,subprocess,numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import emlearn
for k in (2,3):
  X,y = make_classification(n_samples=3000,n_features=12,n_informative=8,n_classes=k,random_state=0)
  X=StandardScaler().fit(X[:2000]).transform(X).astype(np.float32)
  m=RandomForestClassifier(n_estimators=10,max_depth=8,random_state=42).fit(X[:2000],y[:2000])
  Xt=X[2000:]; ref=m.predict(Xt)
  hard=np.array([np.bincount([int(t.predict(Xt[i:i+1])[0]) for t in m.estimators_],minlength=k).argmax() for i in range(len(Xt))])
  print(k,'python hard-vote vs predict', np.mean(hard==ref))
  for lb in (0,8):
    c=emlearn.convert(m,method='inline',dtype='float',leaf_bits=lb); c.save(file=f'k{k}/rf_lb.h',name='model')
    open(f'k{k}/rl.c','w').write('#include <stdio.h>\n#include "rf_lb.h"\nint main(){FILE*f=fopen("X.bin","rb");float x[12];while(fread(x,4,12,f)==12)printf("%d\\n",(int)model_predict(x,12));}')
    r=subprocess.run(f'clang -O2 -w -I{os.path.dirname(emlearn.__file__)} rl.c -o rl && ./rl',shell=True,cwd=f'k{k}',capture_output=True,text=True)
    p=np.array(r.stdout.split(),int); print(k,'emlearn leaf_bits',lb,'agree predict',np.mean(p==ref),'agree hard',np.mean(p==hard), r.stderr[:200])
