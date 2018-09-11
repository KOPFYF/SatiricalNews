# test re
import re
text = 'a'
re.findall(r'(abc)*',text)
print('a'+'f'+'c')

c = []
if not c:
	print('aaaa')

url = 'https://www.cnn.com/2018/06/24/sport/gallery/what-a-shot-sports-0625/index.html'
print('gallery' in url)

txt = 'whaha spost-<a href="https://www.cnn.com/2018/06/20/politics/peter-fonda-baron-trump-secret-service/index.html">Peter Fonda tweet</a>, pre-jacket.'
para = re.sub(r"<a\shref=.*?>", "", txt)
para = re.sub(r"</a>", "", para)
print(para)