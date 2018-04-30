import os
import numpy

#filename2score = lambda x: x[:x.rfind('.')].split('_')

img_files = sorted(os.listdir('samples'))

with open('images.txt') as fid:  
    content = fid.read()  
    content = content.split('\n')    
    content = content[:-1]
    with open('val.txt', 'w') as test_txt:
    	for f in content[:]:
        	ids,name=f.split(' ')
        	cla,a,b=name.split('.')
        	line = '{} images/{}\n'.format(cla,name)
        	test_txt.write(line)
