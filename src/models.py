import numpy as np

from src.utils import *

import requests
import os

from sklearn.preprocessing import RobustScaler, LabelEncoder

import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model, to_categorical

import tensorflow as tf

tf.config.run_functions_eagerly(True)


class Quiniela:
    def __init__(self, model1=None, model2=None, dobles=2):
        self.model1 = model1 if model1 is not None else QuinielaFillerKeras(liga=1, metrics=['accuracy'])
        self.model2 = model2 if model2 is not None else QuinielaFillerKeras(liga=2, metrics=['accuracy'])
        self.dobles = dobles

        if self.model1 is None:
            self.model1.train(batch_size=8)
        if self.model2 is None:
            self.model2.train(batch_size=8)

    def predict_quiniela_pct(self, pleno_al_15=1, *args):
        q1 = self.model1.predict_quiniela_pct(*args)
        q2 = self.model2.predict_quiniela_pct(*args)
        if pleno_al_15 == 1:
            q = pd.concat([q1.iloc[:-1, :], q2], ignore_index=True)
            q = q.append(q1.iloc[-1, :], ignore_index=True)
        else:
            q = pd.concat([q1, q2.iloc[:-1, :]], ignore_index=True)
            q = q.append(q2.iloc[-1, :], ignore_index=True)
        q.index = range(1, 16)
        # Change column name from HDA to 1X2
        q[["1", "X", "2"]] = q[["H", "D", "A"]]
        q.drop(["H", "D", "A"], axis=1, inplace=True)
        return q

    def predict_quiniela(self, pleno_al_15=1, *args):
        pct = self.predict_quiniela_pct(pleno_al_15=1, *args)
        dobles = self.predict_dobles(pct)
        # Choose best option for each fixture
        pct["1X2"] = pct[["1", "X", "2"]].idxmax(axis=1)
        # Drop unused columns
        pct.drop(["1", "X", "2"], axis=1, inplace=True)
        pct["Dobles"] = dobles
        return pct

    def predict_dobles(self, pct):
        preds_pct = pct.copy()
        col = preds_pct[["1", "X", "2"]].idxmax(axis=1)
        for j in range(1, 16):
            preds_pct.loc[j, col[j]] = 0
        dobles = pd.Series(np.repeat("-", 15), index=range(1, 16))
        for i in range(self.dobles):
            max_col = preds_pct[["1", "X", "2"]].max().argmax()
            max_row = preds_pct[["1", "X", "2"]].max(axis=1).argmax()
            preds_pct.iloc[max_row, max_col + 3] = 0  # + 3 for columns Date, HT, AT
            dobles[max_row + 1] = ["1", "X", "2"][max_col]
        return dobles


class QuinielaFillerBase:
    def __init__(self, liga=1, update_data=False):
        if update_data:
            self.update_csvs()

        self.raw_data = prepare_hist_data(liga=liga)
        self.X, self.Y = self.preprocessing()
        self.liga = liga

    def create_model(self):
        pass

    def train(self):
        pass

    def predict(self, xt):
        pass

    def pretty_predict(self, xt, *args):
        preds = self.predict(xt, *args)
        preds_df = pd.DataFrame(preds, columns=self.encoder.classes_)
        return pd.concat([xt[["time", "home_team", "away_team"]], preds_df[["H", "D", "A"]]], axis=1)

    def predict_quiniela_pct(self, pretty=True, *args):
        live_data = prepare_live_data(liga=self.liga)
        quiniela = prepare_quiniela()
        quiniela_live = pd.merge(live_data, quiniela, how="inner")
        if pretty:
            return self.pretty_predict(quiniela_live, *args)
        else:
            return self.predict(quiniela_live, *args)

    def evaluate(self, xt, yt):
        pass

    def save(self):
        pass

    def load(self):
        pass

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
    def __init__(self, liga=1, update_data=False,
                 loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy']):
        super(QuinielaFillerKeras, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(loss=loss, optimizer=optimizer, metrics=metrics)

    def create_model(self,
                     loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy']):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_shape=(self.X.shape[1],)))
        model.add(Dropout(.1))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(.1))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model

    # Trains model from historic_data inputted in initialization
    # + Predictions + Past odds
    def train(self, epochs=200, batch_size=5, verbose=1, early_stopping=True, callbacks=[]):
        if early_stopping:
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            callbacks.append(callback)
        return self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                              callbacks=callbacks)

    def predict(self, xt, batch_size=16, verbose=1):
        categorical = ["home_team", "away_team", "time"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = xt.drop(columns=categorical + unnecessary)
        df_num[["full_time_result.1", "full_time_result.2", "full_time_result.X"]] = 1 / df_num[
            ["full_time_result.1", "full_time_result.2", "full_time_result.X"]]
        xt_scaled = self.rob_scaler.transform(df_num)
        return self.model.predict(xt_scaled, batch_size=batch_size, verbose=1)

    def visualize_model(self):
        self.model.summary()
        plot_model(self.model, to_file='/tmp/model.png', show_shapes=True, )

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)

    def save(self, path="models/", name="model"):
        self.model.save(path + name)

    def load(self, path="models/", name="model"):
        self.model = keras.models.load_model(path + name)


# TODO: Random Forest
class QuinielaFillerRF(QuinielaFillerBase):
    pass


# TODO: Hierarchical clustering
class QuinielaFillerHC(QuinielaFillerBase):
    pass


# TODO: SVM model
class QuinielaFillerSVM(QuinielaFillerBase):
    pass


# TODO: LSTM model
class QuinielaFillerLSTM(QuinielaFillerBase):
    pass
