import sys,subprocess,numpy as np
sys.setrecursionlimit(100000)
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import m2cgen
X,y = make_classification(n_samples=3000,n_features=12,n_informative=8,n_classes=3,random_state=0)
X=StandardScaler().fit(X[:2000]).transform(X).astype(np.float32)
for shape in ('ovr','ovo'):
  m=SVC(kernel='rbf',gamma=1/12,decision_function_shape=shape).fit(X[:2000],y[:2000])
  open('k3/s.h','w').write(m2cgen.export_to_c(m))
  open('k3/s.cpp','w').write('#include <stdio.h>\n#include <string.h>\n#include <math.h>\n#include "s.h"\nint main(){FILE*f=fopen("X.bin","rb");float x[12];double xd[12],o[3];while(fread(x,4,12,f)==12){for(int j=0;j<12;j++)xd[j]=x[j];score(xd,o);printf("%.6f %.6f %.6f\\n",o[0],o[1],o[2]);}}')
  subprocess.run('clang++ -O2 -w s.cpp -o s',shell=True,cwd='k3',check=True)
  out=np.loadtxt(subprocess.run('./s',cwd='k3',capture_output=True,text=True,shell=True).stdout.splitlines())
  df=m.decision_function(X[2000:]); ref=m.predict(X[2000:])
  print(shape,'maxabs diff vs decision_function', np.abs(out-df).max(), 'argmax agree', np.mean(out.argmax(1)==ref), 'sk argmax(df)==predict', np.mean(df.argmax(1)==ref))
pairs=[(0,1),(0,2),(1,2)]
v=np.zeros((len(out),3))
for c,(i,j) in enumerate(pairs):
  v[:,i]+= out[:,c]>0; v[:,j]+= out[:,c]<=0
print('ovo voting agree', np.mean(v.argmax(1)==ref))
