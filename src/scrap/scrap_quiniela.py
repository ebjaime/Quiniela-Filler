import requests
from bs4 import BeautifulSoup

def scrap_todays_quiniela():
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:104.0) Gecko/20100101 Firefox/104.0'}

    url = "https://www.mundodeportivo.com/servicios/quiniela"

    session = requests.Session()

    page = session.get(url, headers=headers)

    soup = BeautifulSoup(page.content, "html.parser")

    scrap_quiniela = soup.find("section", id="quiniela", class_="content").find_all("div", class_="row-cell1")

    quiniela=[]
    for fixt in scrap_quiniela:
        fixture = fixt.find("div", class_="bg-name").text
        quiniela.append(fixture)
    pleno_15 = soup.find("section", id="quiniela", class_="content").find_all("div", class_="row-last1")
    quiniela.append(pleno_15[0].find("div", class_="bg-name").text + " - " + pleno_15[1].find("div", class_="bg-name").text)
    return quiniela