import pandas as pd
from soccerapi.api import Api888Sport
from soccerapi.api import ApiUnibet
# from soccerapi.api import ApiBet365


def scrap_live_odds():
    apis = [Api888Sport(), ApiUnibet()]

    odds_primera = []
    odds_segunda = []
    for api in apis:
        comps = api.competitions()

        url_primera = comps["Spain"]["La Liga"]
        url_segunda = comps["Spain"]["La Liga 2"]

        odds_primera.append(api.odds(url_primera))
        odds_segunda.append(api.odds(url_segunda))

    return odds_primera, odds_segunda

