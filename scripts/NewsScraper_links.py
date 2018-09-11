import urllib
import urllib.request
from urllib.request import urlopen, Request
import re
from bs4 import BeautifulSoup, SoupStrainer
import sys
import datetime
import time
from lxml import etree
import os


'''
UK True:
https://www.theguardian.com/uk-news
https://www.bbc.co.uk/news/uk
https://news.sky.com/uk
https://www.telegraph.co.uk/news/uk/
https://www.express.co.uk/news/uk
https://www.thesun.co.uk/news/

UK Satire:
daily mash


AU Satire:
http://www.theshovel.com.au/

'''


class News:
    def __init__(self, name, url, url0, category, date, country_path, re_pattern):
        self.name = name
        self.url = url
        self.url0 = url0  # 'http://www.abc.net.au'
        # self.urls = urls
        self.category = category
        self.date = date
        self.country_path = country_path
        self.re_pattern = re_pattern
        # self.links = links

    def IsConnectionFailed(self, url):
        """
        check url validity
        """

        try:
            url = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            urlopen(url)
        except (urllib.request.HTTPError, urllib.error.URLError, urllib.error.HTTPError) as error:
            # print('Could not load page')
            return False
        return True

        # try:
        #     urlopen(url)
        # except urllib.request.HTTPError as err:
        #     print('Could not load page')
        #     if err.code == 404:
        #         return False
        #     else:
        #         raise
        # return True

    def url2urls(self):
        '''
        input: domain of News
        output: all category sources
        '''
        temp_urls = [self.url] * len(self.category)
        self.urls = list(
            map(lambda x: x[0] + x[1], zip(temp_urls, self.category)))
        self.urls.append(url)
        print(self.urls)
        return self.urls

    def get_links(self):
        self.links = set()
        for url in self.urls:
            if self.IsConnectionFailed(url) == True:
                url = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urlopen(url)
                # else:
                #     url = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                #     resp = urlopen(url)
                soup = BeautifulSoup(
                    resp, "lxml", from_encoding=resp.info().get_param('charset'))
                for link in soup.find_all('a', href=True):
                    link_str = link['href']
                    if re.match(self.re_pattern, link_str):
                        # print(link_str,type(link_str))
                        # print('******')
                        # if link_str[0] == '/':  # ABC/BBC pattern
                        if link_str[0] == 'h':  # DM pattern
                            link_str = self.url0 + link_str
                            self.links.add(link_str)
                            print(link_str)
                            print('******')
        print('valid links:', len(self.links))
        # print(self.links)
        return self.links

    def linkinlink(self, last_link_set):
        this_link_set = set()
        for url in last_link_set:
            if self.IsConnectionFailed(url) == True:
                url = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urlopen(url)
                soup = BeautifulSoup(
                    resp, "lxml", from_encoding=resp.info().get_param('charset'))
                for link in soup.find_all('a', href=True):
                    link_str = link['href']
                    if re.match(self.re_pattern, link_str):
                        # print(link_str,type(link_str))
                        # print('******')
                        # link_str = 'http://www.abc.net.au' + link_str
                        # if link_str[0] == '/':
                        if link_str[0] == 'h':  # DM pattern
                            link_str = self.url0 + link_str
                            # print(link_str)
                            # print('******')
                            this_link_set.add(link_str)
                print('valid links in this loop', len(this_link_set))
                self.links = self.links.union(this_link_set)
        return this_link_set

    def writelinks(self):
        self.unionlinks = self.links
        print('total links in ', self.name, ':', len(self.unionlinks))
        linkfile_path = os.path.join(
            self.country_path, self.name + '_' + self.date + '.txt')
        # print(linkfile_path)
        with open(linkfile_path, 'w') as f:
            for link in self.unionlinks:
                f.writelines(link + '\n')
        return self.unionlinks


if __name__ == "__main__":
    year = datetime.datetime.today().year
    month = datetime.datetime.today().month
    day = datetime.datetime.today().day
    date = str(year) + '_' + str(month) + '_' + str(day)

    uk_true_path = 'links/uk/true'
    uk_fake_path = 'links/uk/fake'
    au_true_path = 'links/au/true'
    au_fake_path = 'links/au/fake'

    # name = 'ABC'
    # url = "http://www.abc.net.au/news/"
    # url0 = "http://www.abc.net.au"
    # category = ['world', 'science', 'sport', 'politics', 'business',
    #             'health', 'arts-culture', 'analysis-and-opinion']
    # # category = ['world']
    # re_pattern = r".*?/201"
    # ABC = News(name, url, url0, category, date, au_true_path, re_pattern)
    # ABC.url2urls()
    # ABC.get_links()
    # ABC.linkinlink()
    # ABC.linkinlink2()
    # ABC.writelinks()

    name = 'DailyMash'
    url = "https://www.thedailymash.co.uk/"
    url0 = ""
    dailymash_category = ['news', 'sport', 'politics',
                          'opinion', 'features/agony-aunt', 'features/horoscopes']
    # dailymash_category = ['sport']
    re_pattern = r"https://www.thedailymash.co.uk.*?-20"
    DM = News(name, url, url0, dailymash_category,
              date, uk_fake_path, re_pattern)
    DM.url2urls()
    links1 = DM.get_links()
    for i in range(30):
        links1 = DM.linkinlink(links1)
        print('DM links:', len(DM.links))
        if len(DM.links) > 1000:
            break
    DM.writelinks()

    # name = 'BBC'
    # url = "https://www.bbc.com/news/"
    # url0 = "https://www.bbc.com"
    # BBC_category = ['uk', 'world', 'world/us_and_canada', 'business', 'technology', 'science_and_environment',
    #                 'stories', 'entertainment_and_arts', 'health']
    # re_pattern = r".*?-[0-9]+$"
    # BBC = News(name, url, url0, BBC_category, date, uk_true_path, re_pattern)
    # BBC.url2urls()
    # links1 = BBC.get_links()
    # for i in range(30):
    #     links1 = BBC.linkinlink(links1)
    #     print('BBC links:', len(BBC.links))
    #     if len(BBC.links) > 10000:
    #         break
    # BBC.writelinks()
