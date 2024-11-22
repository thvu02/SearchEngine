from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
from bs4 import BeautifulSoup
import regex as re
from pymongo import MongoClient
import pprint

class Frontier:
    def __init__(self, baseurl):
        self.frontier = [baseurl] if baseurl is not None else []

    def done(self):
        return len(self.frontier) == 0
    
    def nextURL(self):
        return self.frontier.pop(0)
    
    def addURL(self, url):
        self.frontier.append(url)
    
class Crawler:
    def __init__(self, baseurl):
        self.frontier = Frontier(baseurl)
        self.visited = set()
        db = self.connectToMongoDB()
        self.pages = db['pages']
    
    def connectToMongoDB(self):
        # connect to local server
        DB_NAME = "CPP"
        DB_HOST = "localhost"
        DB_PORT = 27017
        try:
            client = MongoClient(host=DB_HOST, port=DB_PORT)
            db = client[DB_NAME]
            return db
        except:
            print("Database not connected successfully")
        # delete all documents currently in the collection
        self.pages.delete_many({})

    def debug(self):
        for document in self.pages.find():
            pprint.pprint(document)

    def retrievHTML(self, url):
        try:
            html = urlopen(url)
        except HTTPError as e:
            print(e)
        except URLError as e:
            print('The server could not be found!')
        except ValueError as e:
            print('The URL is not formatted correctly!')
        else:
            self.visited.add(url)
            return html

    def storePage(self, url, html):
        # isTarget will be changed to True via flagTargetPage() if the page meets the criteria
        entry = {
            'url': url,
            'isTarget': False,
            'html': html,
        }
        self.pages.insert_one(entry)
    
    def parseForCriteria(self, bs):
        if bs.find('div', {'class': 'fac-info'}) == None:
            return False
        else:
            return bs.find('div', {'class': 'fac-info'})
    
    def target_page(self, parseForCriteria):
        return parseForCriteria

    def flagTargetPage(self, url):
        self.pages.update_one({'url': url}, {'$set': {'isTarget': True}})
        print(f"Target page found at {url}")

    def clear_frontier(self):
        self.frontier = Frontier(None)

    def parseForLinks(self, bs):
        # find all links with .html or .shtml'))
        discovered_links = [item['href'] for item in bs.find_all('a', href=re.compile(r'^(?!#).*$'))]
        # change all relative links into full addresses
        for i, item in enumerate(discovered_links):
            # CASE: relative address
            if item.startswith('/'):
                discovered_links[i] = "https://www.cpp.edu" + item
            # CASE: full address
            elif item.startswith('http'):
                continue # do nothing
        return discovered_links
    
    def crawl(self, num_targets = 10):
        targets_found = 0
        while not self.frontier.done():
            url = self.frontier.nextURL()
            html = self.retrievHTML(url)
            if html is None:
                continue
            bs = BeautifulSoup(html, 'html.parser')            
            self.storePage(url, bs.prettify())
            if self.target_page(self.parseForCriteria(bs)):
                self.flagTargetPage(url)
                targets_found += 1
            if targets_found == num_targets:
                self.clear_frontier()
                print(f"{targets_found}, targets found. Frontier cleared. Terminating crawl.")
            else:
                for item in self.parseForLinks(bs):
                    if item not in self.visited:
                        self.frontier.addURL(item)

if __name__ == '__main__':
    crawler = Crawler('https://www.cpp.edu/sci/biological-sciences/index.shtml')
    crawler.crawl()
    print("Crawling completed!")