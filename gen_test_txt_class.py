import os
import numpy

#filename2score = lambda x: x[:x.rfind('.')].split('_')

img_files = sorted(os.listdir('samples'))


with open('val.txt', 'w') as test_txt:
    for f in img_files[:]:
        s,score1= f[:f.rfind('.')].split('_')
        line = 'samples/{} {}\n'.format(f,float(score1)/1280)
        test_txt.write(line)