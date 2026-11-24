import subprocess,numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import micromlgen
for k in (2,3):
  X,y=make_classification(n_samples=3000,n_features=12,n_informative=8,n_classes=k,random_state=0)
  X=StandardScaler().fit(X[:2000]).transform(X).astype(np.float32); Xt=X[2000:]
  m=RandomForestClassifier(n_estimators=10,max_depth=8,random_state=42).fit(X[:2000],y[:2000])
  hard=np.array([np.bincount([int(t.predict(Xt[i:i+1])[0]) for t in m.estimators_],minlength=k).argmax() for i in range(len(Xt))])
  open(f'k{k}/rm.h','w').write(micromlgen.port(m,classname='Model'))
  open(f'k{k}/rm.cpp','w').write('#include <stdio.h>\n#include <stdint.h>\n#include "rm.h"\nint main(){Eloquent::ML::Port::Model m;FILE*f=fopen("X.bin","rb");float x[12];while(fread(x,4,12,f)==12)printf("%d\\n",m.predict(x));}')
  p=np.array(subprocess.run('clang++ -O2 -w rm.cpp -o rm && ./rm',shell=True,cwd=f'k{k}',capture_output=True,text=True).stdout.split(),int)
  print(k,'micromlgen RF vs hard vote',np.mean(p==hard),'vs predict',np.mean(p==m.predict(Xt)))
