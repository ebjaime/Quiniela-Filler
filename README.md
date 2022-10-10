# Quiniela-Filler

![](https://upload.wikimedia.org/wikipedia/commons/5/51/La_Quiniela.jpg)

La Quiniela is the name of a Spanish game of chance, managed by Loterías y Apuestas del Estado, which is based on the First and Second Divisions of football. 

This program uses information from different sources (betting odds, market value, past results...) to obtain possible outcomes for the current week's fixtures.

Data sources for this project includes:
* [FiveThirtyEight soccer predictions](https://data.fivethirtyeight.com/#soccer-spi)
* [Football-Data.co.uk](https://www.football-data.co.uk/spainm.php)
* [Transfermarkt data](https://www.transfermarkt.com/primera-division/startseite/wettbewerb/ES1)
* [soccerapi](https://github.com/S1M0N38/soccerapi)

## Example
```
from src.models import QuinielaFillerGBC, QuinielaFillerXGBoost


qf1 = QuinielaFillerXGBoost(liga=1, n_estimators=200, update_data=True)

qf2 = QuinielaFillerGBC(liga=2, n_estimators=200)

qf1.train()
qf2.train()
qf1.save()
# qf1.load("loquesea")

qf = Quiniela(qf1, qf2)
preds = qf.predict_quiniela()
print(preds)
```
From where we expect an output similar to:
```
1   08/10/2022       ALM       RVA   2      -
2   08/10/2022       ATM       GIR   1      -
3   08/10/2022       SEV       ATH   2      1
4   08/10/2022       GET       RMA   2      -
5   09/10/2022       VAD       BET   2      -
6   09/10/2022       CAD       ESP   2      -
7   09/10/2022       RSO       VIL   X      -
8   09/10/2022       BAR       CEL   1      X
9   08/10/2022       PON       GRA   2      -
10  08/10/2022       HUE       LUG   X      -
11  08/10/2022       LPA       IBI   1      -
12  09/10/2022       ZAR       OVI   X      -
13  09/10/2022       LEV       RCG   1      -
14  09/10/2022       ALB       TEN   1      -
15  10/10/2022       ELC       MAL   2      -
```

## Telegram Bot
A telegram bot has been deployed for public use: [Telegram bot](https://t.me/quiniela_filler_bot?)
