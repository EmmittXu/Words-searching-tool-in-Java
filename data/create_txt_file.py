import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

i=104104
while(i>0):
    filename=id_generator()
    filename+='.txt'

    with open('C:\\Users\\EmmittXu\\Desktop\\text.txt', 'r') as f:
        lines = f.readlines()
        #lines = [l for l in lines]
        with open("C:\\Users\\EmmittXu\\Desktop\\Data\\" + filename, "w") as f1:
            f1.writelines(lines)
    i-=1

