import glob
import sys
import datetime
import time
import os


def readlinks(linkfile_path):
    with open(linkfile_path, 'r') as f:
        links = f.readlines()
    return links


def merge_links(list_of_files, outlinkfile_path):
    print(list_of_files)
    print(outlinkfile_path)
    merge_links = set()
    for file in list_of_files:
        links = readlinks(file)
        for link in links:
        	if link:
            	merge_links.add(link)

    with open(outlinkfile_path, 'w') as f:
        for link in merge_links:
            f.writelines(link)
    print('merge_links:',merge_links)
    return merge_links


if __name__ == '__main__':
    year = datetime.datetime.today().year
    month = datetime.datetime.today().month
    day = datetime.datetime.today().day
    date = str(year) + '_' + str(month) + '_' + str(day)
    name = 'DailyMash'

    # list_of_files = glob.glob(os.path.join('/homes/fei8/scratch/SpaceDomain_DATA/links','uk/fake/DailyMash*.txt'))
    list_of_files = glob.glob(os.path.join(
        '/Users/feiyifan/Desktop/NLP/FEEL/scraper', 'links/uk/fake/DailyMash*.txt'))
    
    outlinkfile_path = os.path.join(
        'links/uk/fake', name + '_' + date + '.txt')

    merge_links(list_of_files, outlinkfile_path)
