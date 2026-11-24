import re,subprocess,numpy as np,sys
sys.setrecursionlimit(100000)
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import m2cgen
X,y=make_classification(n_samples=3000,n_features=12,n_informative=8,n_classes=3,random_state=0)
X=StandardScaler().fit(X[:2000]).transform(X).astype(np.float32); Xt=X[2000:]
def tofloat(s):
    s=re.sub(r'\bdouble\b','float',s); s=re.sub(r'\bexp\(','expf(',s); s=re.sub(r'\bpow\(','powf(',s)
    s=re.sub(r'(\d\.\d+(?:[eE][-+]?\d+)?)(?![\dfF])',r'\1f',s); return s
pairs=[(0,1),(0,2),(1,2)]
for name,m in [('rf',RandomForestClassifier(n_estimators=10,max_depth=8,random_state=42)),('svc',SVC(gamma=1/12)),('xgb',XGBClassifier(n_estimators=50,max_depth=4,base_score=0.5,num_parallel_tree=1,random_state=42))]:
  m.fit(X[:2000],y[:2000]); ref=m.predict(Xt); src=m2cgen.export_to_c(m)
  for var,s,T in (('double',src,'double'),('float',tofloat(src),'float')):
    open('k3/f.h','w').write(s)
    open('k3/f.c','w').write('#include <stdio.h>\n#include <string.h>\n#include <math.h>\n#include "f.h"\nint main(){FILE*f=fopen("X.bin","rb");float x[12];%s xd[12],o[3];while(fread(x,4,12,f)==12){for(int j=0;j<12;j++)xd[j]=x[j];score(xd,o);printf("%%f %%f %%f\\n",o[0],o[1],o[2]);}}'%T)
    r=subprocess.run('clang -O2 -Wall -Wdouble-promotion f.c -o f 2>&1 | grep -c double-promotion; ./f',shell=True,cwd='k3',capture_output=True,text=True)
    lines=r.stdout.splitlines(); o=np.loadtxt(lines[1:])
    if name=='svc':
      v=np.zeros((len(o),3))
      for c,(i,j) in enumerate(pairs): v[:,i]+=o[:,c]>0; v[:,j]+=o[:,c]<=0
      p=v.argmax(1)
    else: p=o.argmax(1)
    print(name,var,'agree=%.2f%%'%(100*np.mean(p==ref)),'double-promotion warnings:',lines[0], 'remaining "double":',len(re.findall(r'\bdouble\b',s)))
