import os
with open('inst.txt', 'a+') as f:
    if os.stat('inst.txt').st_size:
        f.seek(0)
        f.truncate()
        f.write('shut up!')
    else:
        print('didn\'t find anything')

