import urllib
import urllib.request
from urllib.request import urlopen
import re
from bs4 import BeautifulSoup, SoupStrainer
import httplib2
import sys
from lxml import etree
# https://pythonspot.com/extract-links-from-webpage-beautifulsoup/


spoof_category = ['us','world','entertainment-gossip','sport','business',
        'science-technology','magazine','hot-topics','uk']
cnn_category = ['us','world','politics','opinions','health',
        'entertainment','style','travel','videos']


def IsConnectionFailed(url):
    """
    check url validity
    """
    try:
        urlopen(url)
    except urllib.request.HTTPError:
        return False
    return True


def GenSpoofLinks(urls):
    for url in urls:
        if IsConnectionFailed(url) == True:
            resp = urlopen(url)
            soup = BeautifulSoup(resp, "lxml",from_encoding=resp.info().get_param('charset'))
            links = set()       
            for link in soup.find_all('a', href=True):
                link_str = link['href']
                # print(link_str)
                for cate in spoof_category:
                    if re.match(r"/spoof-news/"+cate+"/d*?(.+)",link_str):
                        print(link_str)
                        print('******')
                        links.add(link_str)
            print('valid links:', len(links))
            with open('Spoof_satirelinks_708.txt','w') as f:
                for link in links:
                    f.writelines(link+'\n')
    return links


def GenCNNLinks(urls):
    links = set()
    for url in urls:        
        if IsConnectionFailed(url) == True:
            resp = urlopen(url)
            soup = BeautifulSoup(resp, "lxml",from_encoding=resp.info().get_param('charset'))               
            for link in soup.find_all('a', href=True):
                link_str = link['href']
                # print(link_str)
                if re.match(r"/2018",link_str):
                    print(link_str,type(link_str))
                    print('******')
                    links.add(link_str)
            print('valid links:', len(links))
            print(links)

    with open('Cnn_links_708.txt','w') as f:
        for link in links:
            f.writelines(link+'\n')
    return links


# def Loop_LinkinLink(urls):

#     links = list(GenSpoofLinks(urls))
#     pres = ["https://www.thespoof.com"]*len(links)
#     links = list(map(lambda x: x[0]+x[1], zip(pres, links)))
#     links = list(GenSpoofLinks(links))
#     print(len(links))
#     with open('Spoof_satirelinks.txt','w') as f:
#         for link in links:
#             # print(type(link))
#             f.writelines(link+'\n')

urls_spoof = ["https://www.thespoof.com/spoof-news"]*len(spoof_category)
Spoof_urls = list(map(lambda x: x[0]+x[1], zip(urls_spoof, spoof_category)))
Spoof_urls.append("https://www.thespoof.com/")
# print(urls)
# GenSpoofLinks(Spoof_urls)

urls_cnn = ["https://www.cnn.com/"]*len(cnn_category)
Cnn_urls = list(map(lambda x: x[0]+x[1], zip(urls_cnn, cnn_category)))
print(Cnn_urls)
GenCNNLinks(Cnn_urls)

