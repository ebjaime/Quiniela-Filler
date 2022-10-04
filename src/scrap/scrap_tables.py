"""
    Obtains the league's table for the corresponding season to a file. 
    From Transfermarket we can obtain:
        * Total league's table
        * Home / Away league's table
        * Last 5 matches' table
    From fbref.com we obtain the current league's table
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

SEASON = "2022"


# General function for league's tables in tranfermrkt
def scrap_table_transfermrkt(nombres_csv, soups):
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

# Total table
def scrap_total_table_transfermrkt(season=SEASON):
    primera = "https://www.transfermarkt.com/laliga/tabelle/wettbewerb/ES1/saison_id/"+season
    segunda = "https://www.transfermarkt.com/laliga/tabelle/wettbewerb/ES2/saison_id/"+season

    primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
    segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

    primera_soup = BeautifulSoup(primera_page.content, "html.parser")
    segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

    nombres_csv = ["data/tables/primera_total.csv", "data/tables/segunda_total.csv"]
    soups = [primera_soup, segunda_soup]

    primera, segunda = scrap_table_transfermrkt(nombres_csv, soups)
    return primera, segunda

# Home table
def scrap_home_table_transfermrkt(season=SEASON):
    primera = "https://www.transfermarkt.com/laliga/heimtabelle/wettbewerb/ES1/saison_id/"+season
    segunda = "https://www.transfermarkt.com/laliga/heimtabelle/wettbewerb/ES2/saison_id/"+season

    primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
    segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

    primera_soup = BeautifulSoup(primera_page.content, "html.parser")
    segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

    nombres_csv = ["data/tables/primera_home.csv", "data/tables/segunda_home.csv"]
    soups = [primera_soup, segunda_soup]

    primera, segunda = scrap_table_transfermrkt(nombres_csv, soups)
    return primera, segunda


# Away table
def scrap_away_transfermrkt(season=SEASON):
    primera = "https://www.transfermarkt.com/laliga/gasttabelle/wettbewerb/ES1/saison_id/"+season
    segunda = "https://www.transfermarkt.com/laliga/gasttabelle/wettbewerb/ES2/saison_id/"+season

    primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
    segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

    primera_soup = BeautifulSoup(primera_page.content, "html.parser")
    segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

    nombres_csv = ["data/tables/primera_away.csv", "data/tables/segunda_away.csv"]
    soups = [primera_soup, segunda_soup]

    primera, segunda = scrap_table_transfermrkt(nombres_csv, soups)
    return primera, segunda


# Last five
def scrap_last5_transfrmrkt(season=SEASON):
    primera = "https://www.transfermarkt.com/laliga/formtabelle/wettbewerb/ES1"
    segunda = "https://www.transfermarkt.com/laliga/formtabelle/wettbewerb/ES2"

    primera_page = requests.get(primera, headers={'User-Agent': 'Custom'})
    segunda_page = requests.get(segunda, headers={'User-Agent': 'Custom'})

    primera_soup = BeautifulSoup(primera_page.content, "html.parser")
    segunda_soup = BeautifulSoup(segunda_page.content, "html.parser")

    nombres_csv = ["data/tables/primera_last5.csv", "data/tables/segunda_last5.csv"]
    # nombres_csv = ["../historic_data/team_info/primera"+season+".csv", "../historic_data/team_info/segunda"+season+".csv"]
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
    

if __name__ == "__main__":
    scrap_total_table_transfermrkt()
    scrap_home_table_transfermrkt()
    scrap_away_transfermrkt()
    scrap_last5_transfrmrkt()