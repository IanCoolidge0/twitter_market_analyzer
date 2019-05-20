import wx
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from mygrad import sliding_window_view
import tweepy
import datetime
from collections import defaultdict

CONSUMER_TOKEN = "1Vu98GqSeX6FG0bILMJ6lO3qh"
ACCESS_TOKEN = "1858790125-IH9bYyr7sllvPkqOYxTL6dgbWBW1ooDWbh0Uk2C"
CONSUMER_SECRET = ""
ACCESS_SECRET = ""


def poms_scores(text, poms_synonyms):
    raw_scores = defaultdict(int)
    tokens = text.lower().split(' ')

    for token in tokens:
        for word in poms_synonyms:
            if token == word or token in poms_synonyms[word]:
                raw_scores[word] += 1

    scores = dict()

    scores['tension'] = raw_scores['tense'] + \
                        raw_scores['shaky'] + \
                        raw_scores['on edge'] + \
                        raw_scores['panicky'] + \
                       -raw_scores['relaxed'] + \
                        raw_scores['uneasy'] + \
                        raw_scores['restless'] + \
                        raw_scores['nervous'] + \
                        raw_scores['anxious']

    scores['depression'] = raw_scores['unhappy'] + \
                           raw_scores['sorry'] + \
                           raw_scores['sad'] + \
                           raw_scores['blue'] + \
                           raw_scores['hopeless'] + \
                           raw_scores['unworthy'] + \
                           raw_scores['discouraged'] + \
                           raw_scores['lonely'] + \
                           raw_scores['helpless'] + \
                           raw_scores['worthless'] + \
                           raw_scores['terrified'] + \
                           raw_scores['guilty']

    scores['anger'] = raw_scores['angry'] + \
                      raw_scores['peeved'] + \
                      raw_scores['grouchy'] + \
                      raw_scores['spiteful'] + \
                      raw_scores['annoyed'] + \
                      raw_scores['resentful'] + \
                      raw_scores['bitter'] + \
                      raw_scores['ready to fight'] + \
                      raw_scores['rebellious'] + \
                      raw_scores['deceived'] + \
                      raw_scores['furious'] + \
                      raw_scores['bad tempered']

    scores['fatigue'] = raw_scores['worn out'] + \
                        raw_scores['listless'] + \
                        raw_scores['fatigued'] + \
                        raw_scores['exhausted'] + \
                        raw_scores['sluggish'] + \
                        raw_scores['weary'] + \
                        raw_scores['bushed']

    scores['confusion'] = raw_scores['confused'] + \
                          raw_scores['distracted'] + \
                          raw_scores['muddled'] + \
                          raw_scores['bewildered'] + \
                         -raw_scores['efficient'] + \
                          raw_scores['forgetful'] + \
                          raw_scores['uncertain']

    scores['vigour'] = raw_scores['lively'] + \
                       raw_scores['active'] + \
                       raw_scores['energetic'] + \
                       raw_scores['cheerful'] + \
                       raw_scores['alert'] + \
                       raw_scores['peppy'] + \
                       raw_scores['carefree'] + \
                       raw_scores['vigorous']

    return scores


def sentiment_time_series(sent_dict, key):
    N = max(sent_dict.keys())

    T = np.ndarray((N,))
    for i in range(N):
        T[i] = sent_dict[i][key]

    # convert to z-scores
    mean = np.mean(T)
    std = np.std(T)
    T = (T - mean) / std

    return T


def windowed_data(window_size, *ts_data):
    k = window_size * len(ts_data)
    dataset = np.ndarray((len(ts_data[0]) - window_size + 1, k))

    for i, ts in enumerate(ts_data):
        dataset[:, window_size * i:window_size * (i + 1)] = sliding_window_view(ts, (window_size,), 1)

    return dataset[:-1]


def windowed_data_lstm(window_size, *ts_data):
    dataset = np.ndarray((len(ts_data[0]) - window_size + 1, window_size, len(ts_data)))

    for i, ts in enumerate(ts_data):
        dataset[:, :, i] = sliding_window_view(ts, (window_size,), 1)

    return dataset


class MainFrame(wx.Frame):

    def __init__(self, title):
        super().__init__(parent=None, title=title)

        auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

        self.api = tweepy.API(auth)

        self.sent_dict = None
        self.train_djia = None
        self.model_func = None

        self.use_lstm_days = False

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.tweet_filters_label = wx.StaticText(panel, label="Enter filtering condition for tweets:")
        sizer.Add(self.tweet_filters_label, 0, wx.ALL | wx.CENTER, 0)

        self.tweet_filters = wx.TextCtrl(panel)
        sizer.Add(self.tweet_filters, 0, wx.ALL | wx.EXPAND, 10)

        self.load_tweets = wx.Button(panel, label="Load Tweets")
        self.load_tweets.Bind(wx.EVT_BUTTON, self.OnClick)
        sizer.Add(self.load_tweets, 0, wx.ALL | wx.CENTER, 5)

        self.tweet_filters_label = wx.StaticText(panel, label="Choose an algorithm:")
        sizer.Add(self.tweet_filters_label, 0, wx.ALL | wx.CENTER, 10)

        self.alg_selector = wx.ComboBox(panel, choices=["Linear Regression", "Dense Network", "Stacked LSTM"],
                                        value="Stacked LSTM", style=wx.CB_DROPDOWN)
        sizer.Add(self.alg_selector, 0, wx.ALL | wx.CENTER, 0)

        self.train_model = wx.Button(panel, label="Train Model")
        self.train_model.Bind(wx.EVT_BUTTON, self.OnClick)
        sizer.Add(self.train_model, 0, wx.ALL | wx.CENTER, 10)

        self.check_invest = wx.Button(panel, label="Test Buy/Sell")
        self.check_invest.Bind(wx.EVT_BUTTON, self.OnClick)
        sizer.Add(self.check_invest, 0, wx.ALL | wx.CENTER, 50)

        panel.SetSizer(sizer)
        self.SetInitialSize((300, 400))
        self.Show()

    def OnClick(self, event):
        label = event.GetEventObject().GetLabel()

        if label == "Load Tweets":
            self._load_tweets()

        elif label == "Train Model":
            self._train_model()

        elif label == "Test Buy/Sell":
            self._buy_sell()

    def _load_tweets(self):
        with open("djia_ts.txt", "rb") as f:
            self.train_djia = pickle.load(f)[130:130+98]
            self.train_djia = (self.train_djia - np.mean(self.train_djia)) / np.std(self.train_djia)

        with open("sent_dict.txt", "rb") as f:
            self.sent_dict = pickle.load(f)

        with open("poms_synonyms.txt", "rb") as f:
            self.poms_syns = pickle.load(f)

        self.calm = -sentiment_time_series(self.sent_dict, "tension") - sentiment_time_series(self.sent_dict, "anger")
        self.happy = sentiment_time_series(self.sent_dict, "vigour") - sentiment_time_series(self.sent_dict, "depression")
        self.alert = -sentiment_time_series(self.sent_dict, "fatigue") - sentiment_time_series(self.sent_dict, "confusion")

        dfft = np.fft.fft(self.calm)
        dfft[30:] = 0
        self.calm = np.real(np.fft.ifft(dfft))

        dfft = np.fft.fft(self.happy)
        dfft[30:] = 0
        self.happy = np.real(np.fft.ifft(dfft))

        dfft = np.fft.fft(self.alert)
        dfft[30:] = 0
        self.alert = np.real(np.fft.ifft(dfft))

        N = 3
        self.x_train = windowed_data(N + 1, self.calm, self.happy, self.alert)
        self.y_train = self.train_djia[N:-1]

        self.x_train_lstm = windowed_data_lstm(7, self.calm, self.happy, self.alert)
        self.y_train_lstm = self.train_djia[6:]

        print("DJIA and training data loaded.")

    def _buy_sell(self):
        n_days = 3
        if self.use_lstm_days:
            n_days = 7

        day = datetime.datetime.now()

        _calm = np.ndarray((1 + n_days,))
        _happy = np.ndarray((1 + n_days,))
        _alert = np.ndarray((1 + n_days,))

        for i in range(1 + n_days):
            if i == 0:
                tweets = tweepy.Cursor(self.api.search, q="place:96683cc9126741d1").items(100)
            else:
                tweets = tweepy.Cursor(self.api.search, q="place:96683cc9126741d1",
                                       until=day.strftime('%Y-%m-%d')).items(100)
                day -= datetime.timedelta(days=1)

            tweet_count = 0

            for tweet in tweets:
                scores = poms_scores(tweet.text, self.poms_syns)
                _calm[i] += (-scores['tension'] - scores['anger'])
                _happy[i] += (scores['vigour'] - scores['depression'])
                _alert[i] += (-scores['fatigue'] - scores['confusion'])

                tweet_count += 1

            _calm[i] /= tweet_count
            _happy[i] /= tweet_count
            _alert[i] /= tweet_count

        _calm = _calm[::-1]
        _happy = _happy[::-1]
        _alert = _alert[::-1]

        print(_happy)

        print("Prediction: ")
        print(self.model(np.concatenate((_calm, _happy, _alert))))

    def _train_model(self):
        model_type = self.alg_selector.GetValue()

        if model_type == "Linear Regression":
            self.model = self.linear_model
            self.use_lstm_days = False

        elif model_type == "Dense Network":
            self.model = self.nnet_model
            self.use_lstm_days = False

            self.nnet = Sequential()

            self.nnet.add(Dense(512, input_shape=(3 * (3 + 1),), activation="sigmoid"))
            self.nnet.add(Dense(512, activation="sigmoid"))
            self.nnet.add(Dense(1, activation="linear"))

            self.nnet.compile(loss='mse', optimizer='adagrad', metrics=['mse', 'mae'])
            self.nnet.fit(self.x_train, self.y_train, epochs=300, batch_size=10, verbose=0)

        elif model_type == "Stacked LSTM":
            self.model = self.nnet_model
            self.use_lstm_days = True

            self.nnet = Sequential()

            self.nnet.add(LSTM(512, activation="relu", return_sequences=True))
            self.nnet.add(LSTM(512, activation="relu"))
            self.nnet.add(Dense(1, activation="linear"))

            self.nnet.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae'])
            self.nnet.fit(self.x_train_lstm, self.y_train_lstm, epochs=300, batch_size=10, verbose=0)

        print("Model trained.")

    def linear_model(self, x):
        factor = np.linalg.lstsq(self.x_train, self.y_train, rcond=None)[0]
        return np.matmul(x, factor)

    def nnet_model(self, x):
        return self.nnet.predict(x).flatten()


app = wx.App()
frame = MainFrame("Twitter Investing Helper")
app.MainLoop()
