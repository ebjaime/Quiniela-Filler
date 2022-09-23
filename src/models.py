from src.utils import *

from sklearn.preprocessing import RobustScaler, LabelEncoder

import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model, to_categorical

import tensorflow as tf

tf.config.run_functions_eagerly(True)


class QuinielaFiller():
    def __init__(self, liga=1):
        self.raw_data = prepare_hist_data(liga=liga)
        self.X, self.Y = self.preprocessing()
        self.model = self.create_model()

    # Trains model from historic_data inputted in initialization
    # + Predictions + Past odds
    def train(self, epochs=200, batch_size=5, verbose=1):
        return self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, xt, batch_size=5, verbose=1):
        categorical = ["home_team", "away_team", "time"]
        unnecessary = ["total_valA", "total_valH"]

        # REVIEW: should the odds be inverted? To give higher value to favorite result
        df_num = xt.drop(columns=categorical + unnecessary)
        xt_scaled = self.rob_scaler.transform(df_num)
        return self.model.predict(xt_scaled, batch_size=batch_size, verbose=1)

    def pretty_predict(self, xt, batch_size=5, verbose=1):
        preds = self.predict(xt,batch_size,verbose)
        preds_df = pd.DataFrame(preds, columns=self.encoder.classes_)
        return pd.concat([xt[["time", "home_team", "away_team"]], preds_df[["H","D","A"]]], axis=1)

    def evaluate(self, xt, yt):
        return self.model.evaluate(xt, yt, verbose=1)

    def save(self, path="models/", name="model"):
        self.model.save(path + name)

    def load(self, path="models/", name="model"):
        self.model = keras.models.load_model(path + name)

    # TODO: Updates csv for the current season
    def update_data(self):
        pass

    def create_model(self,
                     loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy']):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_shape=(self.X.shape[1],)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        return model

    def preprocessing(self):
        categorical = ["HomeTeam", "AwayTeam", "Date", "FTR", "season"]
        unnecessary = ["total_valA", "total_valH"]

        # REVIEW: should the odds be inverted? To give higher value to favorite result
        df_num = self.raw_data.drop(columns=categorical + unnecessary)
        df_Y = self.raw_data["FTR"]

        # REVIEW: try other scaler for better results?
        self.rob_scaler = RobustScaler().fit(df_num)
        df_rob = self.rob_scaler.transform(df_num)

        self.encoder = LabelEncoder()
        self.encoder.fit(df_Y)
        df_encoded_Y = self.encoder.transform(df_Y)
        dummy_Y = to_categorical(df_encoded_Y, 3)

        return df_rob, dummy_Y

    def visualize_model(self):
        self.model.summary()
        plot_model(self.model, to_file='/tmp/model.png', show_shapes=True, )
