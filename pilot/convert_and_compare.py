import sys, os, re, subprocess, numpy as np
sys.setrecursionlimit(100000)
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import emlearn, m2cgen, micromlgen
EMINC = os.path.dirname(emlearn.__file__)
NF=12
def data(k):
    X,y = make_classification(n_samples=3000,n_features=NF,n_informative=8,n_classes=k,random_state=0)
    sc = StandardScaler().fit(X[:2000]); X = sc.transform(X).astype(np.float32)
    return X[:2000],y[:2000],X[2000:],y[2000:]
G = 1.0/NF  # gamma='scale' on standardized data ~ 1/(n_features*var)=1/NF
models = {
 'dt': lambda: DecisionTreeClassifier(max_depth=5,random_state=42),
 'rf': lambda: RandomForestClassifier(n_estimators=10,max_depth=8,random_state=42),
 'gnb': lambda: GaussianNB(),
 'svc': lambda: SVC(kernel='rbf',gamma=G,random_state=42),
 'xgb': lambda: XGBClassifier(n_estimators=50,max_depth=4,base_score=0.5,random_state=42),
}
def harness(tool, hdr, k, sig):
    if tool=='emlearn':
        call = 'model_predict(x, NF)'
        inc = f'#include "{hdr}"'
        pre = ''
    elif tool=='micromlgen':
        inc = f'#include "{hdr}"'; pre='Eloquent::ML::Port::Model m;'
        call = 'm.predict(x)'
    else:
        inc = f'extern "C" {{\n#include "{hdr}"\n}}' if False else f'#include "{hdr}"'
        pre=''
        if sig=='vec':
            call = f'({{double xd[NF],o[{max(k,2)}]; for(int j=0;j<NF;j++) xd[j]=x[j]; score(xd,o); int b=0; for(int c=1;c<{max(k,2)};c++) if(o[c]>o[b]) b=c; b;}})'
        else:
            call = '({double xd[NF]; for(int j=0;j<NF;j++) xd[j]=x[j]; score(xd)>0?1:0;})'
    return f'''#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#define NF {NF}
{inc}
{pre}
int main(){{ FILE*f=fopen("X.bin","rb"); float x[NF]; while(fread(x,sizeof(float),NF,f)==NF){{ int p={call}; printf("%d\\n",p);}} return 0;}}
'''
for k in (2,3):
  Xtr,ytr,Xte,yte = data(k)
  d=f'k{k}'; os.makedirs(d,exist_ok=True); Xte.tofile(f'{d}/X.bin')
  for name,mk in models.items():
    m = mk().fit(Xtr,ytr); ref = m.predict(Xte)
    extra = f' nSV={m.support_vectors_.shape[0]}' if name=='svc' else ''
    for tool in ('emlearn','micromlgen','m2cgen'):
      hdr=f'{name}_{tool}.h'; out=f'{d}/{hdr}'
      try:
        if tool=='emlearn':
          if name in ('svc','xgb'): continue
          c = emlearn.convert(m, method='inline', dtype='float') if name in('dt','rf') else emlearn.convert(m)
          c.save(file=out, name='model')
        elif tool=='micromlgen':
          if name=='gnb': m.sigma_ = m.var_
          open(out,'w').write(micromlgen.port(m, classname='Model'))
        else:
          if name=='gnb': continue
          open(out,'w').write(m2cgen.export_to_c(m))
      except Exception as e:
        print(f'{k}cls {name:4} {tool:10} CONVERT FAIL {type(e).__name__}: {str(e)[:90]}'); continue
      src=open(out).read()
      sig = 'vec' if 'void score(' in src else 'scalar'
      open(f'{d}/h_{name}_{tool}.cpp','w').write(harness(tool,hdr,k,sig))
      cc = ['clang++','-std=gnu++17','-O2','-w',f'-I{EMINC}',f'h_{name}_{tool}.cpp','-o',f'h_{name}_{tool}']
      r = subprocess.run(cc,cwd=d,capture_output=True,text=True)
      if r.returncode: print(f'{k}cls {name:4} {tool:10} COMPILE FAIL {r.stderr.strip().splitlines()[0][:120]}'); continue
      r = subprocess.run([f'./h_{name}_{tool}'],cwd=d,capture_output=True,text=True)
      pred = np.array([int(v) for v in r.stdout.split()])
      if len(pred)!=len(ref): print(f'{k}cls {name:4} {tool:10} RUN FAIL n={len(pred)}'); continue
      dbl = src.count('double'); flt = src.count('float')
      print(f'{k}cls {name:4} {tool:10} agree={np.mean(pred==ref)*100:6.2f}%  size={len(src)//1024}KB double={dbl} float={flt}{extra}')
