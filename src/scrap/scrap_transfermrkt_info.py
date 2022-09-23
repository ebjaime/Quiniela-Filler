import requests
from bs4 import BeautifulSoup
import pandas as pd

SEASON = "2022"
SEASON_1 = "202223"

primera = "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1/plus/?saison_id="+SEASON
segunda = "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES2/plus/?saison_id="+SEASON

primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

# nombres_csv = ["../../current_data/primera.csv", "../../current_data/segunda.csv"]
nombres_csv = ["../../historic_data/team_info/"+SEASON_1+"_1.csv", "../../historic_data/team_info/"+SEASON_1+"_2.csv"]
soups = [primera_soup, segunda_soup]

for soup, nombre_csv in zip(soups,nombres_csv):
    results_odd = soup.find(id="yw1").find_all("tr", class_="odd")
    results_even = soup.find(id="yw1").find_all("tr", class_="even")
    results = results_even+results_odd

    laliga = pd.DataFrame(columns=["name", "num_squad", "mean_age", "num_foreign", "mean_val(mill)", "total_val(mill)"])
    team = results[0]
    for team in results:
        name = team.find("td", class_="hauptlink").text[:-1]
        squad_age_foreign = [i.text for i in team.find_all("td", class_="zentriert")[1:]]
        mean_total_market = [i.text[1:-1] for i in team.find_all("td", class_="rechts")]
        laliga = laliga.append(pd.Series([name]+squad_age_foreign+mean_total_market, index=laliga.columns), ignore_index=True)
        print("Inserted "+name)
    laliga.to_csv(nombre_csv,index=False)