import urllib
import urllib.request
from urllib.request import urlopen, Request
import re
from bs4 import BeautifulSoup
import datetime
import time
import sys
from lxml import etree
import io
import os
import time


class News_generator:
    def __init__(self, name, date, links_path, bs4_parser, news_path):
        self.name = name
        self.date = date
        self.links_path = links_path
        self.news_path = news_path
        # self.para_pattern = para_pattern #
        self.bs4_parser = bs4_parser  # "html.parser"

    def IsConnectionFailed(self, url):
        """
        check url validity
        """

        try:
            url = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
            urlopen(url)
            print('Connectioned!')
            # urlopen(url, timeout= 5)
        # except urllib.request.HTTPError:
        except (urllib.request.HTTPError, urllib.error.URLError, urllib.error.HTTPError) as error:
            print('Failed for error:',error)
            return False
        return True

    def CreateDoc(self, url):
        # print('link:',url)
        self.doc = []
        if self.IsConnectionFailed(url) == True:
            url = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
            resp = urlopen(url)
            page = resp.read()
            soup = BeautifulSoup(page, bs4_parser, from_encoding='utf-8')

            # ABC pattern
            # try:
            #     a_div = soup.find('div', {'class': "c75l"})
            #     kk = a_div.findAll('p', attrs={'class': None})
            # except AttributeError:
            #     # a_div = soup.find('div',{'class':"clo-lg-9 col-md-8"})
            #     a_div = soup.find('div', {'class': "row"})
            #     kk = a_div.findAll('p', attrs={'class': None})

            # BBC pattern
            # kk = soup.findAll('p')

            # DM pattern
            a_div = soup.find('div',{'id':"post-content"})
            kk = a_div.findAll('p')

            ss = list(kk)
            # self.doc = []
            for s in ss:
                para = re.findall(r'<p>(.*?)</p>', str(s))
                if para:
                    para = para[0]
                    # print(para)
                    para = re.sub(r"<span.*?>", "", para)
                    para = re.sub(r"</?strong>", "", para)
                    para = re.sub(r"</?small>", "", para)
                    para = re.sub(r"</?sml>", "", para)
                    para = re.sub(r"</?span>", "", para)
                    para = re.sub(r"</?em>", "", para)
                    para = re.sub(r"</?i>", "", para)
                    para = re.sub(r"</?u>", "", para)
                    para = re.sub(r"\xa0", "", para)
                    para = re.sub(r"</?b>", "", para)
                    para = re.sub(r"<br/>", "", para)
                    para = re.sub(r"</?a>", "", para)
                    para = re.sub(r"<a\shref=.*?>", "", para)
                    para = re.sub(r"<script.*?>", "", para)
                    para = re.sub(r"</?script>", "", para)
                    para = re.sub(r"<iframe.*?>", "", para)
                    para = re.sub(r"</?iframe>", "", para)
                    para = re.sub(r"<a\sclass=.*?>", "", para)
                    para = re.sub(r"<a\salt=.*?>", "", para)
                    para = re.sub(r"<svg.*?>", "", para)
                    para = re.sub(r"<path.*?>", "", para)
                    para = re.sub(r"</?svg>", "", para)
                    para = re.sub(r"</?path>", "", para)
                    para = re.sub(r"<img.*?>", "", para)
                    if para != ' ':
                        # BBC redundant
                        # if not re.match(r"Follow.*?Twitter", para) and not re.match(r"Reporting by", para) and not re.search(r"@BBC", para) and not \
                        #         re.match(r"Next story:", para) and not re.match(r"Use #NewsfromElsewhere", para) and not re.match(r"Source: BBC", para) and not\
                        #         re.match(r"Edited by", para) and not re.match(r"Filmed and Directed by", para) and not re.match(r"Join the conversation", para):
                        if para: # DM
                            self.doc.append(para)
        # print(self.doc)
        return self.doc

    def CreateDocs(self):
        linkfile_path = os.path.join(
            self.links_path, self.name + '_' + self.date + '.txt')
        print('linkfile_path:', linkfile_path)
        with open(linkfile_path, 'r', encoding='utf-8') as f:
            links = f.readlines()
        links = set(links)
        # print(links)
        print('Number of ', self.name, ' links:', len(links))
        self.docs = []
        for link in links:
            # link = link[:-1]
            # print('links:', link)
            # print(self.IsConnectionFailed(link))
            # if self.IsConnectionFailed(link) == True:
            print('Valid links:', link)
            doc = self.CreateDoc(link)
            self.docs.append(doc)

        newsfile_path = os.path.join(
            self.news_path, self.name + '_' + self.date + '.txt')
        valid_doc = 0
        with open(newsfile_path, 'w', encoding='utf-8') as f:
            for doc in self.docs:
                if doc:
                    valid_doc += 1
                    for para in doc:
                        if para:
                            f.writelines(para + '\n')
                    f.writelines('******\n')
        print('valid docs:', valid_doc)
        return self.docs


if __name__ == "__main__":
    year = datetime.datetime.today().year
    month = datetime.datetime.today().month
    day = datetime.datetime.today().day
    date = str(year) + '_' + str(month) + '_' + str(day)

    links_uk_true_path = 'links/uk/true'
    links_uk_fake_path = 'links/uk/fake'
    links_au_true_path = 'links/au/true'
    links_au_fake_path = 'links/au/fake'

    news_uk_true_path = 'news/uk/true'
    news_uk_fake_path = 'news/uk/fake'
    news_au_true_path = 'news/au/true'
    news_au_fake_path = 'news/au/fake'

    # name = 'ABC'
    # bs4_parser = "html.parser"

    # ABC = News_generator(name, date, links_au_true_path,
    #                      bs4_parser, news_au_true_path)
    # ABC.CreateDocs()

    name = 'DailyMash'
    # bs4_parser = "html5lib"
    # bs4_parser = "html.parser"
    bs4_parser = "lxml"

    DM = News_generator(name, date, links_uk_fake_path,
                         bs4_parser, news_uk_fake_path)
    DM.CreateDocs()

    # name = 'BBC'
    # bs4_parser = "html.parser"

    # BBC = News_generator(name, date, links_uk_true_path,
    #                      bs4_parser, news_uk_true_path)
    # BBC.CreateDocs()
