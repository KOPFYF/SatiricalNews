import urllib2
import bs4, re
import datetime, time

def IsConnectionFailed(url):
    """
    check url validity
    """
    try:
        urllib2.urlopen(url)
    except urllib2.HTTPError, e:
        return False
    return True

def GenOnionLinks():
    f = open('satirelinks.txt','w')
    # print "Enter the URL you wish to crawl.."
    # print 'Usage  example:- "http://phocks.org/stumble/creepy/" <-- With the double quotes'
    # input("@> ")
    n = 0
    preurl = "http://www.theonion.com"
    while True:
        n += 1
        ourl = "http://www.theonion.com/channels/politics/?page=" + str(n)
        if IsConnectionFailed(ourl) == False: break
        page = urllib2.urlopen(ourl).read()
        # articlelist = bs4.SoupStrainer("div", class_="article-list")
        soup = bs4.BeautifulSoup(page)
        extract = soup.find_all("article", "article")
        # print type(extract)
        # print(extract[0])
        for each in extract:
            
            dt = each.span.string.strip().replace(",", "")
            aurl = preurl+each.a['href']#.strip("http://")
            curday = time.strptime(str(dt),'%b %d %Y')
            #print curday[:6]
            # print each.a['href'].split('/')[1]
            if each.a['href'].split('/')[1] == 'articles':
                f.write(dt + ": " + aurl.strip("http://") + "\n")
        if datetime.datetime(* curday[:6]) < datetime.datetime(2010,4,30): break
        print n
        # for link in extract.find_all('a'):
    print "********\nonionLinkfile generated\n********"
    f.close()

def GenCnnLinks():
    f = open('normallinks.txt','w')
    # print "Enter the URL you wish to crawl.."
    # print 'Usage  example:- "http://phocks.org/stumble/creepy/" <-- With the double quotes'
    # input("@> ")
    n = 0
    preurl = "http://politicalticker.blogs.cnn.com/"
    while True:
        n += 1
        ourl = "http://politicalticker.blogs.cnn.com/page/" + str(n)
        if IsConnectionFailed(ourl) == False: break
        page = urllib2.urlopen(ourl).read()
        # articlelist = bs4.SoupStrainer("div", class_="article-list")
        soup = bs4.BeautifulSoup(page)
        extract = soup.find_all('div', attrs={"class": "cnnPostWrap cnn_wh1600_post_separator"})
        # print extract[0].h2.a['href']
        # dt = extract[0].find('div', attrs={"class": "cnnBlogContentDateHead"}).string.strip().replace(",", "").replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        
        # print time.strptime("October 1 2014",'%B %d %Y')[:6]
        # break
        # print(extract[0])  
        for each in extract:
            
            dt = each.find('div', attrs={"class": "cnnBlogContentDateHead"}).string.strip().replace(",", "").replace("1st", "1").replace("2nd", "2").replace("3rd", "3").replace("th", "")
            aurl = each.h2.a['href']#.strip("http://")
            # print aurl
            curday = time.strptime(str(dt),'%B %d %Y')
            #print curday[:6]
            # print each.a['href'].split('/')[1]

            # if each.a['href'].split('/')[1] == 'articles':
            f.write(dt + ": " + aurl.replace("http://", "") + "\n")
        if datetime.datetime(* curday[:6]) < datetime.datetime(2010,4,30): break
        print n
        # for link in extract.find_all('a'):
    print "********\ncnnLinkfile generated\n********"
    f.close()

def main():
    
    GenOnionLinks()
    GenCnnLinks()
    


if __name__ == "__main__":
    main()
