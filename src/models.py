import numpy as np

from src.utils import *

import requests
import os

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC
from sklearn.utils import shuffle

import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SpatialDropout1D
from keras.utils import plot_model, to_categorical

from xgboost import XGBClassifier
import xgboost

import tensorflow as tf

tf.config.run_functions_eagerly(True)


class Quiniela:
    def __init__(self, model1=None, model2=None, dobles=2):
        self.model1 = model1 if model1 is not None else QuinielaFillerKeras(liga=1, update_data=True,
                                                                            metrics=['accuracy'])
        self.model2 = model2 if model2 is not None else QuinielaFillerKeras(liga=2, update_data=True,
                                                                            metrics=['accuracy'])
        self.dobles = dobles

        if self.model1 is None:
            self.model1.train(batch_size=8)
        if self.model2 is None:
            self.model2.train(batch_size=8)

    def predict_quiniela(self, pleno_al_15=1, **kwargs):
        pct = self.predict_quiniela_pct(pleno_al_15=1, **kwargs)
        dobles = self.predict_dobles(pct)
        # Choose best option for each fixture
        pct["1X2"] = pct[["1", "X", "2"]].idxmax(axis=1)
        # Drop unused columns
        pct.drop(["1", "X", "2"], axis=1, inplace=True)
        pct["Dobles"] = dobles
        return pct

    def predict_quiniela_pct(self, pleno_al_15=1, **kwargs):
        q1 = self.model1.predict_quiniela_pct(**kwargs)
        q2 = self.model2.predict_quiniela_pct(**kwargs)
        if pleno_al_15 == 1:
            q = pd.concat([q1.iloc[:-1, :], q2], ignore_index=True)
            q = q.append(q1.iloc[-1, :], ignore_index=True)
        else:
            q = pd.concat([q1, q2.iloc[:-1, :]], ignore_index=True)
            q = q.append(q2.iloc[-1, :], ignore_index=True)
        q.index = range(1, q.shape[0] + 1)
        # Change column name from HDA to 1X2
        q[["1", "X", "2"]] = q[["H", "D", "A"]]
        q.drop(["H", "D", "A"], axis=1, inplace=True)
        return q

    def predict_dobles(self, pct):
        preds_pct = pct.copy()
        col = preds_pct[["1", "X", "2"]].idxmax(axis=1)
        for j in range(1, pct.shape[0] + 1):
            preds_pct.loc[j, col[j]] = 0
        dobles = pd.Series(np.repeat("-", 15), index=range(1, 16))
        for i in range(self.dobles):
            max_col = preds_pct[["1", "X", "2"]].max().argmax()
            max_row = preds_pct[["1", "X", "2"]].max(axis=1).argmax()
            preds_pct.iloc[max_row, max_col + 3] = 0  # + 3 for columns Date, HT, AT
            dobles[max_row + 1] = ["1", "X", "2"][max_col]
        return dobles

    def get_quiniela(self):
        return scrap_todays_quiniela()


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

    def predict(self, xt, **kwargs):
        categorical = ["home_team", "away_team", "time"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = xt.drop(columns=categorical + unnecessary)
        df_num[["AvgH", "AvgA", "AvgD"]] = 1 / df_num[
            ["full_time_result.1", "full_time_result.2", "full_time_result.X"]]
        df_num.drop(["full_time_result.1", "full_time_result.2", "full_time_result.X"], axis=1, inplace=True)
        xt_scaled = self.rob_scaler.transform(df_num[self.columns])  # Apply transformer + Resort columns

        preds = self.model.predict_proba(xt_scaled, **kwargs)
        return preds

    def pretty_predict(self, xt, **kwargs):
        preds = self.predict(xt, **kwargs)
        preds_df = pd.DataFrame(preds, columns=self.encoder.classes_)
        return pd.concat([xt[["time", "home_team", "away_team"]], preds_df[["H", "D", "A"]]], axis=1)

    def predict_quiniela_pct(self, pretty=True, **kwargs):
        live_data = prepare_live_data(liga=self.liga)
        quiniela = prepare_quiniela()
        quiniela_live = pd.merge(live_data, quiniela, how="inner")
        if pretty:
            return self.pretty_predict(quiniela_live, **kwargs)
        else:
            return self.predict(quiniela_live, **kwargs)

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
        self.raw_data = self.raw_data.sample(frac=1)
        categorical = ["HomeTeam", "AwayTeam", "Date", "FTR", "season"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = self.raw_data.drop(columns=categorical + unnecessary)
        df_num[["AvgH", "AvgA", "AvgD"]] = 1 / df_num[["AvgH", "AvgA", "AvgD"]]
        df_Y = self.raw_data["FTR"]

        self.rob_scaler = RobustScaler()
        df_rob = self.rob_scaler.fit_transform(df_num)
        self.columns = df_num.columns

        self.encoder = LabelEncoder()
        self.encoder.fit(df_Y)
        df_encoded_Y = self.encoder.transform(df_Y)
        dummy_Y = to_categorical(df_encoded_Y, 3)

        return df_rob, dummy_Y


class QuinielaFillerKeras(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerKeras, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self,
                     loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'],
                     dropout=True):
        model = Sequential()
        model.add(Dense(16, activation=None, input_shape=(self.X.shape[1],)))
        if dropout:
            model.add(Dropout(.5))
        model.add(Dense(8, activation='relu'))
        if dropout:
            model.add(Dropout(.5))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model

    # Trains model_nn from historic_data inputted in initialization
    # + Predictions + Past odds
    def train(self, epochs=50, batch_size=16, verbose=1, early_stopping=True, callbacks=[]):
        if early_stopping:
            callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, min_delta=0.01)
            callbacks.append(callback)
        return self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                              callbacks=callbacks)

    def predict(self, xt, batch_size=16, verbose=1):
        categorical = ["home_team", "away_team", "time"]
        unnecessary = ["total_valA", "total_valH"]

        df_num = xt.drop(columns=categorical + unnecessary)
        df_num[["AvgH", "AvgA", "AvgD"]] = 1 / df_num[
            ["full_time_result.1", "full_time_result.2", "full_time_result.X"]]
        df_num.drop(["full_time_result.1", "full_time_result.2", "full_time_result.X"], axis=1, inplace=True)
        xt_scaled = self.rob_scaler.transform(df_num[self.columns])  # Apply transformer + Resort columns

        return self.model.predict(xt_scaled, batch_size=batch_size, verbose=1)

    def visualize_model(self):
        self.model.summary()
        plot_model(self.model, to_file='/tmp/model_nn.png', show_shapes=True, )

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)

    def save(self, path="models/", name="model_nn"):
        self.model.save(path + name)

    def load(self, path="models/", name="model_nn"):
        self.model = keras.models.load_model(path + name)


class QuinielaFillerRF(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerRF, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, criterion='entropy', n_estimators=100, max_depth=16, warm_start=True, verbose=1):
        model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth,
                                       warm_start=warm_start,
                                       verbose=verbose,
                                       random_state=42)
        return model

    def train(self):
        self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)


class QuinielaFillerKNN(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerKNN, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, n_neighbors):
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)


class QuinielaFillerQDA(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerQDA, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, priors=None, reg_param=0.0):
        return QDA(priors=priors, reg_param=reg_param)

    def train(self):
        self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)


class QuinielaFillerSVM(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerSVM, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, C=1.0, kernel="rbf"):
        return SVC(C=C, kernel=kernel, probability=True)

    def train(self):
        self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)


class QuinielaFillerLSTM(QuinielaFillerKeras):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerLSTM, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self,
                     loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'],
                     dropout=True):
        model = Sequential()
        model.add(SpatialDropout1D(0.1, input_shape=(16, 1)))
        model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model


class QuinielaFillerGBC(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerGBC, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, learning_rate=0.1, n_estimators=100, verbose=1):
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, verbose=verbose)
        return model

    def train(self):
        self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)


class QuinielaFillerXGBoost(QuinielaFillerBase):
    def __init__(self, liga=1, update_data=False, **kwargs):
        super(QuinielaFillerXGBoost, self).__init__(liga=liga, update_data=update_data)
        self.model = self.create_model(**kwargs)

    def create_model(self, n_estimators=100):
        model = XGBClassifier(n_estimators=n_estimators)
        return model

    def train(self):
        # self.Y = self.raw_data["FTR"]
        self.model.fit(self.X, self.Y)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt)

    def save(self, name="model_xgb"):
        self.model.save_model("models/" + name + ".json")

    def load(self, name="model_xgb"):
        self.model = xgboost.XGBClassifier()
        self.model.load_model("models/" + name + ".json")
