from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib.request
from bs4 import BeautifulSoup
import time
import concurrent.futures

MAX_THREADS = 30
url = "https://edition.cnn.com"
topic = '/politics'

urlTarget = url + topic

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

CHROME_DRIVER_PATH = "C:/Users/YASSER/Downloads/chromedriver.exe"

driver = webdriver.Chrome(CHROME_DRIVER_PATH, chrome_options=options)

data = []

def getLinks():
    driver.get(urlTarget)
    time.sleep(10)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    allarticles = soup.find_all("article")
    links = []
    for singleArt in allarticles:
        links.append(url + singleArt.a.get('href'))
    return links


def getArticle(link):
    articleLink = urllib.request.urlopen(link)

    soup = BeautifulSoup(articleLink, "html.parser")

    title = soup.find_all("h1", {"class": "pg-headline"})[0].get_text()

    author = soup.find_all(attrs={"class": "metadata__byline__author"})[0].get_text()

    processing = []

    paragraphs = soup.find_all(attrs={"class": "zn-body__paragraph"})
    for p in paragraphs:
        processing.append(p.get_text())

    article = ' '.join(processing)
    data.append(
        {
            "title": title,
            "author": author,
            "link": link,
            "body": article
        })


def getArticles(links):
    threads = min(MAX_THREADS, len(links))

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(getArticle, links)


def mainScraping(links):
    t0 = time.time()
    getArticles(links)
    t1 = time.time()
    print(f"{t1 - t0} seconds to download {len(links)} articles.")
    return data



# if __name__ == "__main__":
#    links=getLinks()
#    mainScraping(links)