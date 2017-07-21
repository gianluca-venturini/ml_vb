import itertools
from scipy import misc
from PIL import Image
from PIL import ImageChops
import os
import glob

from text_ovelapping_model.params import window_size, step_size

sort = ['good100']
path = '../../../../ran/Documents/ML_bad_lf/overlap/'

#Given a photo we produce a smaller windows
def crop(p_name,s):
    if p_name == ".DS_Store":
        return
    try:
        img = Image.open(path + s + '/' + p_name)
        n, m = img.size
        print n,m,p_name,
        for i, j in itertools.product(range(0, n-window_size, step_size), range(0, m-window_size, step_size)):
            print '.',
            cropped_img = img.crop((i, j, i+window_size, j+window_size))
            if ImageChops.invert(cropped_img).getbbox():
                cropped_img.save(path + 'cropped-' + s + '/' + p_name + '-' + str(i) + '-' + str(j) + '.png')
        print '\n'
    except:
        pass


#preparing photo windows from given photos. prints the name of each file and each dot represents a cropped photo
k=0
for i in sort:
    listing = os.listdir(path + i)
    for infile in listing:
        k+=1
        print k,
        crop(infile,i)