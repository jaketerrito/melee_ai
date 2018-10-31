from PIL import Image

import numpy as np
import glob

res = {}
for f in glob.glob('./*.jpg'):
	bw = np.asarray(Image.open(f).convert('L'))[13	,45]
	name = f[f.index('\\')+1:f.index('.j')]
	res[name] = bw

for i in range(0,290):
	print(res[str(i)])
