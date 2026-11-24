import sys,os,subprocess,numpy as np, warnings; warnings.filterwarnings('ignore')
sys.setrecursionlimit(100000)
import xgboost, m2cgen, micromlgen
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
D=sys.argv[1]; os.makedirs(D,exist_ok=True)
for k in (2,3):
  X,y = make_classification(n_samples=3000,n_features=12,n_informative=8,n_classes=k,random_state=0)
  X=StandardScaler().fit(X[:2000]).transform(X).astype(np.float32); Xt=X[2000:]; Xt.tofile(f'{D}/X.bin')
  for bs in (0.5,):
    kw={'base_score':bs,'num_parallel_tree':1}
    m=XGBClassifier(n_estimators=50,max_depth=4,random_state=42,**kw).fit(X[:2000],y[:2000]); ref=m.predict(Xt)
    for tool in ('m2cgen','micromlgen'):
      tag=f'xgb{xgboost.__version__} k={k} base_score={bs} {tool:10}'
      try:
        if tool=='m2cgen':
          src=m2cgen.export_to_c(m); open(f'{D}/x.h','w').write(src)
          nout=max(k,2)
          main=('double xd[12],o[%d];for(int j=0;j<12;j++)xd[j]=x[j];score(xd,o);int b=0;for(int c=1;c<%d;c++)if(o[c]>o[b])b=c;printf("%%d\\n",b);'%(nout,nout))
        else:
          src=micromlgen.port(m,classname='Model'); open(f'{D}/x.h','w').write(src)
          main='static Eloquent::ML::Port::Model mm; printf("%d\\n",mm.predict(x));'
      except Exception as e:
        print(tag,'CONVERT FAIL',type(e).__name__,str(e)[:80]); continue
      open(f'{D}/x.cpp','w').write('#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n#include <math.h>\n#include "x.h"\nint main(){FILE*f=fopen("X.bin","rb");float x[12];while(fread(x,4,12,f)==12){'+main+'}}')
      r=subprocess.run('clang++ -std=gnu++17 -O2 -w x.cpp -o x && ./x',shell=True,cwd=D,capture_output=True,text=True)
      if r.returncode: print(tag,'COMPILE/RUN FAIL',r.stderr[:120]); continue
      p=np.array(r.stdout.split(),int); nan='NaN-in-code' if 'nan' in src.lower().replace('nan','',0) and ' nan' in src.lower() else ''
      print(tag,'agree=%.2f%%'%(100*np.mean(p==ref)), nan)
