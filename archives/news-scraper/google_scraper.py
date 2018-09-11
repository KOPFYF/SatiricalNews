import sys, os
# webdriver is needed to lauch a browser
from selenium import webdriver 
# search for things using specific params
from selenium.webdriver.common.by import By 
# wait for page to load
from selenium.webdriver.support.ui import WebDriverWait 
# specify conditions that signal the page has been loaded
from selenium.webdriver.support import expected_conditions as EC 
# Handling timeouts
from selenium.common.exceptions import TimeoutException

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from selenium.webdriver.common.keys import Keys

import csv
import time
import datetime
import sys
from bs4 import BeautifulSoup as bs
import re
import argparse
from gen_test_data import gen_data

def config(headless=False):
    options = webdriver.ChromeOptions()

    # path to the chrome executable
    options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'

    # run in headless mode
    if headless:
        options.add_argument('headless')

    # set window size
    options.add_argument('window-size=1200x600')

    # in incognito mode
    options.add_argument(' — incognito')

    '''
    caps = DesiredCapabilities.CHROME
    caps["pageLoadStrategy"] = "none"
    '''

    return options


class Google:
    def __init__(self, options):
        self.driver = webdriver.Chrome(chrome_options=options)
        self.datafile = None
        self.datawriter = None
    

    def find_results(self, driver):
        # urls = driver.find_elements_by_css_selector('h3.r a')
        urls = driver.find_elements_by_xpath('//h3[@class="r"]/a')
        if urls:
            return urls
        else:
            return False


    def search(self, key):
        url = 'https://google.com'

        self.driver.get(url)
        # print search
        time.sleep(10)

        # find the search box
        search_bx = self.driver.find_element_by_name('q')

        # enter our query
        search_bx.send_keys(key)
        url = self.driver.find_element_by_css_selector('h3.r a')
        print(url.get_attribute('href'))
        
        self.driver.quit()


    def handle_error(self, e):
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        self.driver.quit()
        
        self.datafile.close()
        sys.exit(1)
         

    def connect_to_url(self, url, times=0):
        try:
            self.driver.get(url)
            time.sleep(1)
        except Exception as e:
            print('something\'s wrong with driver')
            if times >= 5:
                self.handle_error(e)
            else:
                self.connect_to_url(url, times+1)


    def find_element_by_name(self, name):
        try:
            # find the search box
            return self.driver.find_element_by_name(name)
        except Exception as e:
            print('cannot find', name)
            self.handle_error(e)


    def load(self, timeout, time_elem, wait=0):
        try:
            WebDriverWait(self.driver, timeout).until(\
                    EC.visibility_of_element_located((By.XPATH, time_elem)))
            return True
        except TimeoutException as e:
            print('Timed out')
            if wait >= 5:
                # self.handle_error(e)
                return False
            else:
                return self.load(timeout, time_elem, wait+1)


    def find_element_by_xpath(self, filename, xpath, index=0):
        try:
            url = None
            if not index:
                url = self.driver.find_element_by_xpath(xpath)
            else:
                # refresh the search results
                self.driver.refresh()
                url = self.driver.find_elements_by_xpath(xpath)[index]
            href = url.get_attribute('href')

            print(href)
            
            # if not the desired website, keep trying
            if 'theonion' not in href and 'thespoof' not in href \
                    and 'satireword' not in href and 'thebeaverton' not in href \
                    and 'ossurworld' not in href and 'dailycurrant' not in href and 'nationalreport' not in href \
                    and 'snopes' not in href and 'satirewire' not in href \
                    and 'syruptrap' not in href and 'unconfirmedsource' not in href:
                print('No match, looking for the {} result'.format(index+1))
                return self.find_element_by_xpath(filename, xpath, index+1)
            
            return url, href
        except Exception as e:
            print(e)
            # either not on the first page or no result at all, give up
            print('no result')
            return None, None
        

    def write_data(self, data):
        self.datawriter.writerow(data)
        self.datafile.flush()


    def batch_search(self, generator):
        self.datafile = open('dates.csv', 'a', newline='')
        self.datawriter = csv.writer(self.datafile, delimiter=',')

        i = 0
        for para, filename in generator:
            if i < 0:
                i += 1
                continue

            # using google
            '''
            self.connect_to_url("https://google.com")

            search_bx = self.find_element_by_name('q')
            '''

            # using yahoo
            '''
            self.connect_to_url("https://yahoo.com")

            search_bx = self.find_element_by_name('p')
            '''

            # using bing
            self.connect_to_url("https://bing.com")

            search_bx = self.find_element_by_name('q')

            # format query
            synop = para[:100]
            print('QUERY:', synop, '...')
            para = para[:500] + '\n'

            # clear the textbox
            search_bx.clear()

            # enter our query
            search_bx.send_keys(para)

            # google's configuration
            # url, href = self.find_element_by_xpath(filename, '//h3[@class="r"]/a')

            # yahoo's config
            url, href = self.find_element_by_xpath(filename, '//h2/a')

            if not url and not href:
                self.write_data([synop,None, None])
                continue

            time_elem = None

            if 'theonion' in href:
                time_elem = '//time[@class="meta__time updated"]'
            elif 'thespoof' in href:
                time_elem = '//p[@id="writtenOn"]'
            elif 'satireword' in href:
                time_elem = '//p[@class="post-details"]'
            elif 'thebeaverton' in href:
                time_elem = '//time[@class="time"]'
            elif 'ossurworld' in href:
                time_elem = '//time[@class="entry-date"]'
            elif 'dailycurrant' in href:
                time_elem = '//div[@class="meta"]/span[@class="primary-color-text"]'
            elif 'nationalreport' in href:
                time_elem = '//dive[@class="entry-meta"]/abbr'
            elif 'snopes' in href:
                # satire tribune
                time_elem = '//span[@class="date-wrapper"]'
            elif 'satirewire' in href:
                time_elem = '//span[@class="entry-date"]'
            elif 'syruptrap' in href:
                time_elem = '//div[@class="entry-meta"]'
            elif 'unconfirmedsource' in href:
                time_elem = '//span[@class="entry-meta-date updated"]/a'
            else:
                self.write_data([synop,None,None])
                continue

            try:
                # follow the link
                url.click()
            except Exception as e:
                # if the click fails, try go to the website directly
                print('click failed')
                self.connect_to_url(href)

            # wait for the page to load
            timeout = 20

            loaded = self.load(timeout, time_elem)
            if not loaded:
                self.write_data([synop,href,None])
                continue

            try:
                element = self.driver.find_element_by_xpath(time_elem)
            except Exception as e:
                print('cannot find time')
                self.write_data([synop,href,None])
                continue
            
            datetime = None
            if 'theonion' in href:
                datetime = element.get_attribute('datetime')
            elif 'thespoof' in href:
                datetime = element.text
            elif 'thebeaverton' in href or 'ossurworld' in href:
                datetime = element.get_attribute('datetime')
            elif 'satireworld' in href:
                datetime = element.text
                datetime = datetime[datetime.index('on')+3:]
            elif 'dailycurrant' in href or 'snopes' in href \
                or 'satirewire' in href or 'syruptrap' in href or 'unconfirmedsource' in href:
                datetime = element.text
            elif 'nationalreport' in href:
                datetime = element.get_attribute('title')
            else:
                self.write_data([synop,href,None])
                continue

            print(datetime)
            self.write_data([synop,href,datetime])

        self.driver.quit()
        
        self.datefile.close()


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('in_date', help='inbound date')
    args = parser.parse_args()
    '''

    # options= config()
    options= config(True)
    g = Google(options)
    g.batch_search(gen_data('text/satire/'))
