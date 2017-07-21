import string

# for i in range(1000,1100):
#     s=unichr(i)
#     print s,
from string import lowercase

l_c= list(string.ascii_lowercase)
l_l  = list(string.ascii_letters)
print_non_whitespace = string.digits + string.letters + string.punctuation +lowercase +lowercase+lowercase
l_p = list(print_non_whitespace)
u =u""
# for i in range(0,1100):
#     print unichr(i),


import random

a = ""
for i in range(1,10):
    random. shuffle(l_p)
    for l in l_p:
        a = a+l
print a
#
# random. shuffle(l_c)
# a=""
# for l in l_c:
#     a = a+l
# print a