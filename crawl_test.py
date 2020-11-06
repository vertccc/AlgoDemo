import urllib
from bs4 import BeautifulSoup

# proxy = urllib.request.ProxyHandler({'https':'47.88.71.201:8808'})
# opener = urllib.request.build_opener(proxy)


#url_address = 'https://www.imdb.com/title/tt1070874/?ref_=hm_fanfav_tt_3_pd_fp1'

url_address = 'https://www.imdb.com/title/tt8579674/?ref_=ttls_li_tt'


html = urllib.request.urlopen(url_address).read()

soup = BeautifulSoup(html,features="lxml")

rr = soup.find(class_='ratingValue').contents[1]
rate = float(rr.string)
users = rr['title'].split()[3]



title_year = soup.find(class_='title_wrapper').contents[1].getText()
title_year_split = title_year.split()
title = ' '.join(title_year_split[:-1])
year = title_year_split[-1]

print('{} is rated {} based on {} users'.format(title,rate,users))

# all save images in the website
i = 1
for x in soup.findAll('img'):
    url = x['src']
    path = str(i)+'.jpg'
    urllib.request.urlretrieve(url,path)
    i += 1