import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:104.0) Gecko/20100101 Firefox/104.0'}

primera = "https://fbref.com/en/comps/12/La-Liga-Stats"
segunda = "https://fbref.com/en/comps/17/Segunda-Division-Stats"

session = requests.Session()

primera_page = session.get(primera, headers=headers)
segunda_page = session.get(segunda, headers=headers)

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

nombres_csv = ["../../current_data/primera_table.csv", "../../current_data/segunda_table.csv"]
# nombres_csv = ["../../historic_data/team_info/primera"+SEASON+".csv", "../../historic_data/team_info/segunda"+SEASON+".csv"]
soups = [primera_soup, segunda_soup]

for soup, nombre_csv in zip(soups,nombres_csv):
    results = soup.find("table", class_="stats_table").find_all("tr")

    laliga = pd.DataFrame(columns=["Rk","Squad","MP","W","D","L","GF","GA","GD","Pts","Pts/MP","xG","xGA","xGD","xGD/90","Last 5","Attendance"])

    for team in results:
        rk = team.find("th",class_="qualification_indicator").text
        info = [i.text for i in team.find_all("td")[:-3]]
        laliga = laliga.append(pd.Series([rk]+info, index=laliga.columns), ignore_index=True)
    laliga.to_csv(nombre_csv,index=False)