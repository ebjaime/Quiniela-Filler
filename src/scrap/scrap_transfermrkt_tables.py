import requests
from bs4 import BeautifulSoup
import pandas as pd

SEASON = "2022"

def scrap(nombres_csv, soups):
    tables = []
    for soup, nombre_csv in zip(soups,nombres_csv):
        results = soup.find(id="yw1").find_all("tr")[1:]

        laliga = pd.DataFrame(columns=["club","Matches_Played","W","D","L","Goals","+/-","Pts"])
        for team in results:
            name = team.find_all("td", class_="hauptlink")[1].text[1:-1]
            other = [i.text for i in team.find_all("td", class_="zentriert")][1:]
            laliga = laliga.append(pd.Series([name]+other, index=laliga.columns), ignore_index=True)
            print("Inserted "+name)
        laliga.to_csv(nombre_csv, index=False)
        tables.append(laliga)
    return tables

# TOTAL TABLE
primera = "https://www.transfermarkt.com/laliga/tabelle/wettbewerb/ES1/saison_id/"+SEASON
segunda = "https://www.transfermarkt.com/laliga/tabelle/wettbewerb/ES2/saison_id/"+SEASON

primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

nombres_csv = ["../current_data/tables/primera_total.csv", "../current_data/tables/segunda_total.csv"]
# nombres_csv = ["../historic_data/team_info/primera"+SEASON+".csv", "../historic_data/team_info/segunda"+SEASON+".csv"]
soups = [primera_soup, segunda_soup]

primera, segunda = scrap(nombres_csv, soups)

# HOME TABLE
primera = "https://www.transfermarkt.com/laliga/heimtabelle/wettbewerb/ES1/saison_id/"+SEASON
segunda = "https://www.transfermarkt.com/laliga/heimtabelle/wettbewerb/ES2/saison_id/"+SEASON

primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

nombres_csv = ["../../current_data/tables/primera_home.csv", "../../current_data/tables/segunda_home.csv"]
# nombres_csv = ["../historic_data/team_info/primera"+SEASON+".csv", "../historic_data/team_info/segunda"+SEASON+".csv"]
soups = [primera_soup, segunda_soup]

primera, segunda = scrap(nombres_csv, soups)

# AWAY TABLE
primera = "https://www.transfermarkt.com/laliga/gasttabelle/wettbewerb/ES1/saison_id/"+SEASON
segunda = "https://www.transfermarkt.com/laliga/gasttabelle/wettbewerb/ES2/saison_id/"+SEASON

primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

nombres_csv = ["../../current_data/tables/primera_away.csv", "../../current_data/tables/segunda_away.csv"]
# nombres_csv = ["../historic_data/team_info/primera"+SEASON+".csv", "../historic_data/team_info/segunda"+SEASON+".csv"]
soups = [primera_soup, segunda_soup]

primera, segunda = scrap(nombres_csv, soups)


# LAST FIVE
primera = "https://www.transfermarkt.com/laliga/formtabelle/wettbewerb/ES1"
segunda = "https://www.transfermarkt.com/laliga/formtabelle/wettbewerb/ES2"

primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

primera_soup = BeautifulSoup(primera_page.content, "html.parser")
segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

nombres_csv = ["../../current_data/tables/primera_last5.csv", "../../current_data/tables/segunda_last5.csv"]
# nombres_csv = ["../historic_data/team_info/primera"+SEASON+".csv", "../historic_data/team_info/segunda"+SEASON+".csv"]
soups = [primera_soup, segunda_soup]
for soup, nombre_csv in zip(soups, nombres_csv):
    results = soup.find(class_="responsive-table").find_all("tr")[1:]

    laliga = pd.DataFrame(columns=["club", "Matches_Played", "W", "D", "L", "Goals", "+/-", "Pts", "last5"])
    for team in results:
        name = team.find_all("td", class_="hauptlink")[1].text[1:-1]
        other = [i.text for i in team.find_all("td", class_="zentriert")][1:]
        other[-1] = "".join(other[-1].split("\n"))
        laliga = laliga.append(pd.Series([name] + other, index=laliga.columns), ignore_index=True)
        print("Inserted " + name)
    laliga.to_csv(nombre_csv, index=False)