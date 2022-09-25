from src.utils import *

import requests
import os

from sklearn.preprocessing import RobustScaler, LabelEncoder

import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model, to_categorical

import tensorflow as tf

tf.config.run_functions_eagerly(True)


class QuinielaFillerBase():
    def __init__(self, liga=1, update_data=False):
        if update_data:
            self.update_csvs()

        self.raw_data = prepare_hist_data(liga=liga)
        self.X, self.Y = self.preprocessing()
        self.model = self.create_model()

    # Trains model from historic_data inputted in initialization
    # + Predictions + Past odds
    def train(self):
        pass

    def predict(self, xt):
        pass

    def create_model(self):
        pass

    def evaluate(self, xt, yt):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def pretty_predict(self, xt, batch_size=5, verbose=1):
        preds = self.predict(xt, batch_size, verbose)
        preds_df = pd.DataFrame(preds, columns=self.encoder.classes_)
        return pd.concat([xt[["time", "home_team", "away_team"]], preds_df[["H", "D", "A"]]], axis=1)

    # Updates all .csv files of the current season
    # due to changes in odds, dates...
    def update_csvs(self):
        # [1] Update 2022/23 fixtures and odds (https://www.football-data.co.uk/)
        url_1 = "https://www.football-data.co.uk/mmz4281/2223/SP1.csv"
        url_2 = "https://www.football-data.co.uk/mmz4281/2223/SP2.csv"

        r_1 = requests.get(url_1, allow_redirects=True)
        r_2 = requests.get(url_2, allow_redirects=True)

        open('historic_data/fixtures_and_odds/202223_1.csv', 'wb').write(r_1.content)
        open('historic_data/fixtures_and_odds/202223_2.csv', 'wb').write(r_2.content)

        # [2] Update preds FIveThirtyEight
        url_preds_m = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
        url_preds_c = "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv"

        r_preds_m = requests.get(url_preds_m, allow_redirects=True)
        r_preds_c = requests.get(url_preds_c, allow_redirects=True)

        open('preds/spi_matches.csv', 'wb').write(r_preds_m.content)
        open('preds/spi_global_rankings.csv', 'wb').write(r_preds_c.content)

        os.system(
            "echo 'season,date,league_id,league,team1,team2,spi1,spi2,prob1,prob2,probtie,proj_score1,proj_score2,importance1,importance2,score1,score2,xg1,xg2,nsxg1,nsxg2,adj_score1,adj_score2\n' > preds/spi_primera_matches.csv")
        os.system(
            "echo 'season,date,league_id,league,team1,team2,spi1,spi2,prob1,prob2,probtie,proj_score1,proj_score2,importance1,importance2,score1,score2,xg1,xg2,nsxg1,nsxg2,adj_score1,adj_score2\n' > preds/spi_segunda_matches.csv")
        os.system("echo 'rank,prev_rank,name,league,off,def,spi' > preds/spi_primera_rankings.csv")
        os.system("echo 'rank,prev_rank,name,league,off,def,spi' > preds/spi_segunda_rankings.csv")

        os.system(
            "cat preds/spi_global_rankings.csv | grep \"Spanish Primera Division\" >> preds/spi_primera_rankings.csv")
        os.system(
            "cat preds/spi_global_rankings.csv | grep \"Spanish Segunda Division\" >> preds/spi_segunda_rankings.csv")

        os.system("cat preds/spi_matches.csv | grep \"Spanish Primera Division\" >> preds/spi_primera_matches.csv")
        os.system("cat preds/spi_matches.csv | grep \"Spanish Segunda Division\" >> preds/spi_segunda_matches.csv")

        os.system("rm -r preds/spi_global_rankings.csv")
        os.system("rm -r preds/spi_matches.csv")

        # TODO: [3] Current data TransferMrkt

    def preprocessing(self):
        categorical = ["HomeTeam", "AwayTeam", "Date", "FTR", "season"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = self.raw_data.drop(columns=categorical + unnecessary)
        df_num[["AvgH", "AvgA", "AvgD"]] = 1 / df_num[["AvgH", "AvgA", "AvgD"]]
        df_Y = self.raw_data["FTR"]

        # REVIEW: try other scaler for better results?
        self.rob_scaler = RobustScaler().fit(df_num)
        df_rob = self.rob_scaler.transform(df_num)

        self.encoder = LabelEncoder()
        self.encoder.fit(df_Y)
        df_encoded_Y = self.encoder.transform(df_Y)
        dummy_Y = to_categorical(df_encoded_Y, 3)

        return df_rob, dummy_Y


class QuinielaFillerKeras(QuinielaFillerBase):

    # Trains model from historic_data inputted in initialization
    # + Predictions + Past odds
    def train(self, epochs=200, batch_size=5, verbose=1):
        return self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, xt, batch_size=5, verbose=1):
        categorical = ["home_team", "away_team", "time"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = xt.drop(columns=categorical + unnecessary)
        df_num[["full_time_result.1", "full_time_result.2", "full_time_result.X"]] = 1 / df_num[
            ["full_time_result.1", "full_time_result.2", "full_time_result.X"]]
        xt_scaled = self.rob_scaler.transform(df_num)
        return self.model.predict(xt_scaled, batch_size=batch_size, verbose=1)

    def create_model(self,
                     loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy']):
        # REVIEW: try other confg. options
        model = Sequential()
        model.add(Dense(8, activation='relu', input_shape=(self.X.shape[1],)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model

    def visualize_model(self):
        self.model.summary()
        plot_model(self.model, to_file='/tmp/model.png', show_shapes=True, )

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)

    def save(self, path="models/", name="model"):
        self.model.save(path + name)

    def load(self, path="models/", name="model"):
        self.model = keras.models.load_model(path + name)

    def pretty_predict(self, xt, batch_size=5, verbose=1):
        preds = self.predict(xt, batch_size, verbose)
        preds_df = pd.DataFrame(preds, columns=self.encoder.classes_)
        return pd.concat([xt[["time", "home_team", "away_team"]], preds_df[["H", "D", "A"]]], axis=1)


# TODO: Random Forest
class QuinielaFillerRF(QuinielaFillerBase):
    pass


# TODO: SVM model
class QuinielaFillerSVM(QuinielaFillerBase):
    pass


# TODO: LSTM model
class QuinielaFillerLSTM(QuinielaFillerBase):
    pass
