import pandas as pd
from datetime import datetime
import numpy as np
from src.scrap.scrap_live_odds import scrap_live_odds

hist_data_dir = "historic_data"
preds_dir = "preds"


# FIXME: include segunda teams
def translate_to_abrev(name, opt=1):
    # FiveThirtyEight
    dict_1 = {
        'Alavés': "ALA", 'Almeria': "ALM", 'Athletic Bilbao': "ATH",
        'Atletico Madrid': "ATM", 'Barcelona': "BAR", 'Cadiz': "CAD",
        'Celta Vigo': "CEL", 'Deportivo La Coruña': "DEP", 'Eibar': "EIB",
        'Elche': "ELC", 'Espanyol': "ESP", 'Getafe': "GET", 'Girona FC': "GIR",
        'Granada': "GRA", 'Las Palmas': "LPA", 'Leganes': "LEG", 'Levante': "LEV",
        'Mallorca': "MAL", 'Málaga': "MLG", 'Osasuna': "OSA", 'Rayo Vallecano': "RVA",
        'Real Betis': "BET", 'Real Madrid': "RMA", 'Real Sociedad': "RSO",
        'Real Valladolid': "VAD", 'SD Huesca': "HUE", 'Sevilla FC': "SEV",
        'Sporting Gijón': "SPG", 'Valencia': "VAL", 'Villarreal': "VIL",

        'La Hoya Lorca': "LOR", 'AD Alcorcon': "ALC", 'Cordoba': "COR",
        'Numancia': "NUM", 'Real Oviedo': "OVI",
        'Gimnástic Tarragona': "GIM", 'Lugo': "LUG", 'Albacete': "ALB",
        'Cultural Leonesa': "LEO", 'Sevilla Atletico': "SEB",
        'Real Zaragoza': "ZAR",
        'Reus Deportiu': "REU", 'FC Barcelona II': "BAB",
        'Extremadura UD': "EXT", 'Rayo Majadahonda': "MAJ",
        'Racing Santander': "RCG", 'Mirandes': "MIR",
        'SD Ponferradina': "PON", 'Fuenlabrada': "FUE",
        'Castellon': "CAS", 'FC Cartagena': "CAR", 'Logrones': "LOG", 'CD Sabadell': "SAB",
        'Real Sociedad II': "RSB", 'UD Ibiza': "IBI", 'Amorebieta': "AMO", 'Burgos': "BUR",
        'Villarreal B': "VIB", 'FC Andorra': "AND", "Tenerife":"TEN"

    }
    # https://www.football-data.co.uk/
    dict_2 = {
        'Alaves': "ALA", 'Almeria': "ALM", 'Ath Bilbao': "ATH",
        'Ath Madrid': "ATM", 'Barcelona': "BAR", 'Betis': "BET",
        'Cadiz': "CAD", 'Celta': "CEL", 'La Coruna': "DEP", 'Eibar': "EIB",
        'Elche': "ELC", 'Espanol': "ESP", 'Getafe': "GET",
        'Girona': "GIR", 'Granada': "GRA", 'Huesca': "HUE", 'Las Palmas': "LPA",
        'Leganes': "LEG", 'Levante': "LEV", 'Malaga': "MLG", 'Mallorca': "MAL",
        'Osasuna': "OSA", 'Real Madrid': "RMA", 'Sevilla': "SEV", 'Sociedad': "RSO",
        'Sp Gijon': "SPG", 'Valencia': "VAL", 'Valladolid': "VAD", 'Vallecano': "RVA",
        'Villarreal': "VIL",

        'Mirandes': "MIR", 'Alcorcon': "ALC", 'Cordoba': "COR",
        'Gimnastic': "GIM", 'Numancia': "NUM", 'Sevilla B': "SEB", 'Zaragoza': "ZAR",
        'Tenerife': "TEN", 'Lugo': "LUG", 'Oviedo': "OVI", 'Reus Deportiu': "REU",
        'UCAM Murcia': "UCM", 'Lorca': "LOR",
        'Albacete': "ALB", 'Leonesa': "LEO",
        'Barcelona B': "BAB", 'Extremadura UD': "EXT",
        'Rayo Majadahonda': "MAJ", 'Santander': "RCG", 'Ponferradina': "PON",
        'Fuenlabrada': "FUE", 'Leganes': "LEG", 'Castellon': "CAS", 'Cartagena': "CAR",
        'Logrones': "LOG", 'Sabadell': "SAB", 'Sociedad B': "RSB", 'Ibiza': "IBI", 'Eibar': "EIB",
        'Amorebieta': "AMO", 'Burgos': "BUR", 'Villarreal B': "VIB", 'Andorra': "AND"

    }
    # Transfermarkt
    dict_3 = {
        'FC Barcelona': "BAR", 'Sevilla FC': "SEV", 'Villarreal CF': "VIL",
        'Real Sociedad': "RSO", 'Deportivo de La Coruña': "DEP", 'Málaga CF': "MLG",
        'Real Betis Balompié': "BET", 'SD Eibar': "EIB", 'CD Leganés': "LEG",
        'CA Osasuna': "OSA", 'Real Madrid': "RMA", 'Atlético de Madrid': "ATM",
        'Valencia CF': "VAL", 'Athletic Bilbao': "ATH", 'Celta de Vigo': "CEL",
        'Deportivo Alavés': "ALA", 'UD Las Palmas': "LPA", 'RCD Espanyol Barcelona': "ESP",
        'Granada CF': "GRA", 'Sporting Gijón': "SPG", 'Getafe CF': "GET",
        'Girona FC': "GIR", 'Levante UD': "LEV", 'Rayo Vallecano': "RVA",
        'SD Huesca': "HUE", 'Real Valladolid CF': "VAD", 'RCD Mallorca': "MAL",
        'Elche CF': "ELC", 'Cádiz CF': "CAD", 'UD Almería': "ALM",

        'Sevilla Atlético': "SEB", 'Córdoba CF': "COR", 'Cultural Leonesa': "LEO",
        'CD Lugo': "LUG", 'Real Zaragoza': "ZAR", 'CD Numancia': "NUM", 'CD Tenerife': "TEN",
        'CF Reus Deportiu': "REU",
        'FC Barcelona B': "BAB",
        'Gimnàstic de Tarragona': "GIM", 'Real Oviedo': "OVI",
        'Albacete Balompié': "ALB", 'Lorca FC': "LOR", 'AD Alcorcón': "ALC",
        'CF Rayo Majadahonda': "MAJ",
        'Extremadura UD': "EXT", 'SD Ponferradina': "PON",
        'CF Fuenlabrada': "FUE", 'CD Mirandés': "MIR", 'Racing Santander': "RCG",
        'CD Castellón': "CAS", 'CE Sabadell FC': "SAB",
        'FC Cartagena': "CAR", 'UD Logroñés': "LOG",
        'UD Ibiza': "IBI", 'Real Sociedad B': "RSB", 'SD Amorebieta': "AMO", 'Burgos CF': "BUR",
        'Villarreal CF B': "VIB", 'FC Andorra': "AND"
    }
    # soccerapi
    dict_4 = {
        'Almeria': "ALM", 'Athletic Bilbao': "ATH", 'Atlético Madrid': "ATM", 'Cadiz': "CAD",
        'Celta Vigo': "CEL", 'Elche': "ELC", 'Espanyol': "ESP", 'FC Barcelona': "BAR", 'Getafe': "GET",
        'Girona': "GIR", 'Mallorca': "MAL", 'Osasuna': "OSA", 'Rayo Vallecano': "RVA", 'Real Betis': "BET",
        'Real Madrid': "RMA", 'Real Sociedad': "RSO", 'Sevilla': "SEV", 'Valencia': "VAL",
        'Valladolid': "VAD", 'Villarreal': "VIL",

        'Sporting de Gijón': "SPG", 'Huesca': "HUE", 'Mirandés': "MIR", 'Málaga': "MLG",
        'Ponferradina': "PON", 'Burgos': "BUR", 'Lugo': "LUG", 'Eibar': "EIB", 'Albacete Balompié': "ALB",
        'UD Las Palmas': "LPA", 'FC Cartagena': "CAR",
        'UD Ibiza': "IBI", 'Leganés': "LEG", 'Real Zaragoza': "ZAR", 'Villarreal B': "VIB", 'Tenerife': "TEN",
        'Levante': "LEV", 'Real Oviedo': "OVI", 'Racing Santander': "RCG", 'FC Andorra': "AND",
        'Granada': "GRA", 'Alavés': "ALA"

    }
    return dict_1[name] if opt == 1 else (dict_2[name] if opt == 2 else (dict_3[name] if opt == 3 else dict_4[name]))


def clean_end_ws(name):
    while name[-1] == ' ':
        name = name[:-1]
    return name


def flatten_dict(d, sep='.'):
    return pd.json_normalize(d, sep=sep).to_dict(orient='records')


def Y_m_d_to_d_m_Y(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d').strftime('%d/%m/%Y')


def d_m_y_to_d_m_Y(date_str):
    return datetime.strptime(date_str, '%d/%m/%y').strftime('%d/%m/%Y')


def time_to_d_m_Y(date_str):
    return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').strftime('%d/%m/%Y')


def get_avg_odds(fixt_odds):
    df = pd.DataFrame()
    df["AvgH"] = fixt_odds[["B365H", "BWH", "IWH", "LBH", "PSH", "WHH", "VCH"]].mean(axis=1)
    df["AvgD"] = fixt_odds[["B365D", "BWD", "IWD", "LBD", "PSD", "WHD", "VCD"]].mean(axis=1)
    df["AvgA"] = fixt_odds[["B365A", "BWA", "IWA", "LBA", "PSA", "WHA", "VCA"]].mean(axis=1)
    return df


def correct_value_string(string):
    if isinstance(string, str):
        if string[-2:] == "Th":
            return float(string[:-2])/1000
    return float(string)

# Creates a huge CSV with a row for each fixture between two teams [1],
# information about the match (odds, predictions) [2],
# the teams information (economic,spi...) [3],
# and the final result [4]
def prepare_hist_data(liga=1):
    years = ["201617", "201718", "201819", "201920", "202021", "202122", "202223"]
    if liga == 2:
        years = years[1:]
    # [1,2,4] Read from fixture/odds to obtain
    fixt_ls = []
    for year in years:
        if liga == 1:
            fixt_df = pd.read_csv(hist_data_dir + "/fixtures_and_odds/" + year + "_1.csv")
        else:
            fixt_df = pd.read_csv(hist_data_dir + "/fixtures_and_odds/" + year + "_2.csv")
        # Date starting on the 2018 campaign has a different format
        if year == "201617" or year == "201718":
            fixt_df[["Date"]] = fixt_df[["Date"]].applymap(d_m_y_to_d_m_Y)

        fixt_ls.append(fixt_df)
    fixt = pd.concat(fixt_ls, ignore_index=True)
    fixt_1 = fixt[["Date", "HomeTeam", "AwayTeam", "FTR"]]
    fixt_2 = get_avg_odds(fixt[["B365H", "B365D", "B365A",
                                "BWH", "BWD", "BWA",
                                "IWH", "IWD", "IWA",
                                "LBH", "LBD", "LBA",
                                "PSH", "PSD", "PSA",
                                "WHH", "WHD", "WHA",
                                "VCH", "VCD", "VCA"]])

    fixt = fixt_1.join(fixt_2)
    # Seems like matches involving ReusDeportiu do not have betting odds for the year 2019
    # because it got dissolved due to economic problems. Quick solution: discard
    fixt.dropna(inplace=True, axis=0)

    # [2] Get predictions
    if liga == 1:
        preds = pd.read_csv(preds_dir + "/spi_primera_matches.csv")[:fixt.shape[0]]
    else:
        preds = pd.read_csv(preds_dir + "/spi_segunda_matches.csv")[:fixt.shape[0]]
    preds = preds[["season", "date", "team1", "team2", "spi1", "spi2", "prob1", "prob2", "probtie"]]

    # Combine with fixture information
    # Change date format so we have the same in both fixt data and preds
    preds[["date"]] = preds[["date"]].applymap(Y_m_d_to_d_m_Y)
    # Do the same with the team names
    preds[["team1", "team2"]] = preds[["team1", "team2"]].applymap(translate_to_abrev)
    fixt[["HomeTeam", "AwayTeam"]] = fixt[["HomeTeam", "AwayTeam"]].applymap(translate_to_abrev, opt=2)

    # Perform join statement on the two dataframes
    fixt_preds = pd.merge(fixt, preds, how="left",
                          left_on=["Date", "HomeTeam", "AwayTeam"],
                          right_on=["date", "team1", "team2"]).drop(columns=["date", "team1", "team2"])

    # [3] Get team information
    info_ls = []
    season = 2016 if liga == 1 else 2017
    for year in years:
        if liga == 1:
            info_df = pd.read_csv(hist_data_dir + "/team_info/" + year + "_1.csv")
        else:
            info_df = pd.read_csv(hist_data_dir + "/team_info/" + year + "_2.csv")
        info_df["season"] = season
        info_ls.append(info_df)
        season += 1
    tinfo = pd.concat(info_ls, ignore_index=True)
    tinfo[["name"]] = tinfo[["name"]].applymap(clean_end_ws)
    tinfo[["name"]] = tinfo[["name"]].applymap(translate_to_abrev, opt=3)
    tinfo[["mean_val(mill)"]] = tinfo[["mean_val(mill)"]].applymap(correct_value_string)

    # Join with fixture info and preds
    df_aux = pd.merge(fixt_preds, tinfo, how="inner", left_on=["season", "HomeTeam"], right_on=["season", "name"]).drop(
        columns=["name"])
    df_aux.columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'AvgH', 'AvgD', 'AvgA', 'season',
                      'spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'num_squadH',
                      'mean_ageH', 'num_foreignH', 'mean_valH', 'total_valH']
    df_aux = pd.merge(df_aux, tinfo, how="inner", left_on=["season", "AwayTeam"], right_on=["season", "name"]).drop(
        columns=["name"])
    df_aux.columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'AvgH', 'AvgD', 'AvgA', 'season',
                      'spi1', 'spi2', 'prob1', 'prob2', 'probtie',
                      'num_squadH', 'mean_ageH', 'num_foreignH', 'mean_valH', 'total_valH',
                      'num_squadA', 'mean_ageA', 'num_foreignA', 'mean_valA', 'total_valA']

    return df_aux


def prepare_live_data(liga=1):
    # [1] Get odds
    live_1, live_2 = scrap_live_odds()
    live = live_1 if liga == 1 else live_2
    live_l = []
    live_info = None
    for l in live:
        l_aux = flatten_dict(l)
        l_aux = pd.DataFrame(l_aux)
        live_l.append(l_aux[["full_time_result.1", "full_time_result.X", "full_time_result.2"]])
        live_info = l_aux[["time", "home_team", "away_team"]]

    aux = pd.concat(live_l, axis=1)
    aux = aux.groupby(level=0, axis=1).mean() / 1000

    fixt = pd.concat([live_info, aux], axis=1)

    fixt[["home_team", "away_team"]] = fixt[["home_team", "away_team"]].applymap(translate_to_abrev, opt=4)
    fixt[["time"]] = fixt[["time"]].applymap(time_to_d_m_Y)

    # [2] Get preds
    if liga == 1:
        preds = pd.read_csv("preds/spi_primera_matches.csv")
    else:
        preds = pd.read_csv("preds/spi_segunda_matches.csv")
    preds = preds[["date", "team1", "team2", "spi1", "spi2", "prob1", "prob2", "probtie"]]

    # Combine with fixture information
    # Change date format so we have the same in both fixt data and preds
    preds[["date"]] = preds[["date"]].applymap(Y_m_d_to_d_m_Y)
    # Do the same with the team names
    preds[["team1", "team2"]] = preds[["team1", "team2"]].applymap(translate_to_abrev)

    fixt_preds = pd.merge(fixt, preds, how="left",
                          left_on=["time", "home_team", "away_team"],
                          right_on=["date", "team1", "team2"]).drop(columns=["date", "team1", "team2"])
    # [3] team info
    if liga==1:
        tinfo = pd.read_csv("historic_data/team_info/202223_1.csv")
    else:
        tinfo = pd.read_csv("historic_data/team_info/202223_2.csv")
    tinfo[["name"]] = tinfo[["name"]].applymap(clean_end_ws)
    tinfo[["name"]] = tinfo[["name"]].applymap(translate_to_abrev, opt=3)
    tinfo[["mean_val(mill)"]] = tinfo[["mean_val(mill)"]].applymap(correct_value_string)

    # Join with fixture info and preds
    df_aux = pd.merge(fixt_preds, tinfo, how="inner", left_on=["home_team"], right_on=["name"]).drop(
        columns=["name"])
    df_aux.columns = ['time', 'home_team', 'away_team', 'full_time_result.1',
                      'full_time_result.2', 'full_time_result.X', 'spi1', 'spi2', 'prob1',
                      'prob2', 'probtie', 'num_squad', 'mean_age', 'num_foreign',
                      'mean_val', 'total_val']
    df_aux = pd.merge(df_aux, tinfo, how="inner", left_on=["away_team"], right_on=["name"]).drop(
        columns=["name"])
    df_aux.columns = ['time', 'home_team', 'away_team', 'full_time_result.1',
                      'full_time_result.2', 'full_time_result.X', 'spi1', 'spi2', 'prob1',
                      'prob2', 'probtie',
                      'num_squadH', 'mean_ageH', 'num_foreignH', 'mean_valH', 'total_valH',
                      'num_squadA', 'mean_ageA', 'num_foreignA', 'mean_valA', 'total_valA']

    return df_aux


if __name__ == "__main__":
    print(prepare_hist_data())
