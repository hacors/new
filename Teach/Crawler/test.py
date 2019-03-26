from urllib.request import urlopen

html = urlopen(
    "https://morvanzhou.github.io/tutorials/data-manipulation/scraping/1-01-understand-website/"
).read().decode('utf-8')
print(html)
