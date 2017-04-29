import requests
from bs4 import BeautifulSoup
li_links =[]
for i in range(32):
    r = requests.get("https://www.poemhunter.com/emily-dickinson/poems/page-"+str(i)+"/?a=a&l=3&y=")
    soup = BeautifulSoup(r.content,"lxml")
    links = soup.find_all("td",{"class":"title"})
    page_link ="https://www.poemhunter.com"
    for i in links:
        a = i.find('a')
        li_links.append((page_link+a.get('href'),a.get('title')))
#print(li_links)
#print(len(li_links))
for i in range(len(li_links)):
    r = requests.get(li_links[i][0])
    soup = BeautifulSoup(r.content,"lxml")
    poems = soup.find_all("div",{"class":"KonaBody"})
    for j in poems:
    	a = j.find('p')
    	print(li_links[i][1])
    	print("\n")
    	print(a.text.strip())
    	print("\n")