import os
import numpy

#filename2score = lambda x: x[:x.rfind('.')].split('_')

img_files = sorted(os.listdir('samples'))

max=min=0
with open('val.txt', 'w') as test_txt:
    for f in img_files[:]:
        s,score1,score2,score3,score4,score5,score6 = f[:f.rfind('.')].split('_')
        score1=float(score1)/10
        score1=int(score1)
        score1+=123
        if(score1>max):
        	max=score1
        if(score1<min):
        	min=score1
        line = 'samples/{} {}\n'.format(f,score1)
        test_txt.write(line)
print(max,min)