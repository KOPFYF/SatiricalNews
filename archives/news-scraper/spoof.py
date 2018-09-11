import urllib
from urllib.request import urlopen
import re
from bs4 import BeautifulSoup
import datetime, time
import sys
from lxml import etree
import io
# global page


def IsConnectionFailed(url):
    """
    check url validity
    """
    try:
        urlopen(url)
    except urllib.request.HTTPError:
        return False
    return True


def CreateSpoofDoc(url):
	if IsConnectionFailed(url) == True:		
		response = urlopen(url)
		page = response.read()
		soup = BeautifulSoup(page,'html.parser',from_encoding='utf-8')
	
	time_selector = etree.HTML(page)
	time = time_selector.xpath('//p[@id="writtenOn"]')
	# print(time)

	a_div = soup.find('div',{'id':"articlebody"},{'itemprop':"articleBody"})
	kk = a_div.findAll('p')
	# print(kk,type(kk))
	ss = list(kk)
	# print(ss,type(ss),len(ss))
	doc = []
	for s in ss:
		# print(s,type(s))
		para = raw2para(str(s))
		# print(para,type(para))
		# print(para[0][1])
		para_ = re.sub(r"</?em>", "", para[0][1])
		para_ = re.sub(r"</?u>", "", para_)
		para_ = re.sub(r"<br/>", "", para_)
		doc.append(para_)
		# print(doc)
	return doc


def CreateCnnDoc(url):
	if IsConnectionFailed(url) == True:		
		response = urlopen(url)
		page = response.read()
		soup = BeautifulSoup(page,'html.parser',from_encoding='utf-8')

	if 'gallery' in url:
		pass
	else:
		# body = soup.find('div',{'class':"pg-rail-tall__body",'itemprop':"articleBody"})
		body = soup.find('div',{'itemprop':"articleBody"})
		first_para = body.find('p',{'class': "zn-body__paragraph speakable"})
		# speak_div = body.findAll('div',{'class': "zn-body__paragraph speakable"})
		# para_div = body.findAll('div',{'class': "zn-body__paragraph"})
		# read_all = body.find('div',{'class': "zn-body__read-all"})
		# left_div = read_all.findAll('div',{'class': "zn-body__paragraph"})

		paras = soup.findAll('div',{'class': "zn-body__paragraph"})
		# print(paras)

		first_para = re.findall(r'<p.*cite>(.*?)</p>',str(first_para))[0]
		# print(first_para)
		doc = []
		doc.append(first_para)
		for s in paras:
			para = re.findall(r'<div.*">(.*?)</div>',str(s))
			# print(para)
			if para:
				if para[0] != ' ':
					para = re.sub(r"</?em>", "", para[0])
					para = re.sub(r"</?a>", "", para)
					# para = re.sub(r"</a>", "", para)
					para = re.sub(r"</?ul>", "", para)
					para = re.sub(r"</?li>", "", para)
					para = re.sub(r"</?h3>", "", para)
					para = re.sub(r"<a\shref=.*?>", "", para)
					# print(para)
					doc.append(para.strip())
		# for s in speak_div:
		# 	speak_para = re.findall(r'<div.*speakable">(.*?)</div>',str(s))
		# 	# print(speak_para,type(speak_para))
		# 	if speak_para:
		# 		doc.append(speak_para)

		# for s in para_div:
		# 	# print(s)
		# 	para = re.findall(r'<div.*paragraph">(.*?)</div>',str(s))
		# 	print(para,type(para),len(para))
		# 	if para:
		# 		doc.append(para[0])

		# for s in left_div:
		# 	# print(s)
		# 	para = re.findall(r'<div.*paragraph">(.*?)</div>',str(s))
		# 	print(para,type(para),len(para))
		# 	if para:
		# 		doc.append(para[0])
		# print(doc)
		return doc

# CreateCnnDoc('https://www.cnn.com/2018/07/02/opinions/democratic-leaders-theres-an-elephant-in-the-room-bakari-sellers/index.html')
# CreateCnnDoc('https://www.cnn.com/2018/07/06/entertainment/sharp-objects-review/index.html')



def raw2para(text):
	return re.findall(r'<p>(<em>)?(.*?)(</em>)?</p>',text)


def CreateSpoofDocs(out_file_name):
	with open('Spoof_satirelinks.txt','r') as f:
		links = f.readlines() 
	docs = []
	for link in links:
		# link = re.sub(r"/spoof-news", "", link))
		link = 'https://www.thespoof.com'+link
		if IsConnectionFailed(link) == True:
			print('Valid links:',link)
			if 'gallery' not in url:
				doc = CreateSpoofDoc(link)
				docs.append(doc)
		else:
			print('invalid links:',link)

	with io.open(out_file_name,'w',encoding='utf8') as f:
		for doc in docs:
			for para in doc:
				f.writelines(para +'\n')
			f.writelines('******\n')


def CreateCnnDocs(out_file_name):
	with open('Cnn_links_708.txt','r') as f:
		links = f.readlines() 
	docs = []
	for link in links:
		link = 'https://www.cnn.com'+link
		if IsConnectionFailed(link) == True:
			print('Valid links:',link)
			doc = CreateCnnDoc(link)
			docs.append(doc)
		else:
			print('invalid links:',link)

	with io.open(out_file_name,'w',encoding='utf8') as f:
		for doc in docs:
			for para in doc:
				f.writelines(para +'\n')
			f.writelines('******\n')
	
# CreateSpoofDocs('spoof_news.txt')
CreateCnnDocs('Cnn_news.txt')

# url1 = 'https://www.thespoof.com/magazine/14399/born-to-spoof-new-beginnings-ch-4-clone-ride-to-oblivion'
# url2 = 'https://www.thespoof.com/entertainment-gossip/130325/shakespeare-experiment-comes-off-rails-at-eleventh-hour'
# url3 = 'https://www.thespoof.com/science-technology/130283/longest-day-over'
# CreateSpoofDoc(url1)
# CreateSpoofDoc(url3)