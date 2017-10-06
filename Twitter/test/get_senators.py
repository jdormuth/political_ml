from bs4 import BeautifulSoup as bs 
import urlparse
from urllib2 import urlopen
from urllib import urlretrieve
import os
import sys
import pickle


url = 'https://www.thomasmore.org/us-senate-twitter-account-list/'
soup = bs(urlopen(url), 'html.parser')


republican_handles =[]
democratic_handles = []
for parent in soup.findAll('tbody', "row-hover"):
   
    for child in parent.children:
        try:
            l = child.contents
            if len(l[2].contents) > 0:
                party = l[2].contents[0].encode('ascii','ignore')
                handle = l[4].contents[0].encode('ascii','ignore')
                handle = handle.split(' & ')
               
                if(party == "Republican"):
                    republican_handles += handle
                    
                if(party == "Democratic"):
                    democratic_handles += handle
                    

        except AttributeError:
            

print(republican_handles)
print(democratic_handles)

pickle.dump(republican_handles, open('republicans','w'))
pickle.dump(democratic_handles, open('democrats', 'w'))
quit()
print(soup.findAll('td'))



