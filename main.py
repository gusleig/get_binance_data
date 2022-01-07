from tqdm import tqdm
import pandas as pd
from binance.client import Client
import sqlalchemy
import numpy as np
import os
import logging
import threading
import sys
import itertools
# import talib as ta
import matplotlib.pyplot as plt
import ta.momentum
import ta.trend
import ta.volatility
import api_keys

client = Client(api_keys.api_key, api_keys.api_secret)

logger = logging.getLogger('SimpleDCABot')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AccountBag:

    def __init__(self, coin='USDT', init_balance=0):

        self.wallet = pd.DataFrame(columns=['Balance'])

        self.wallet.loc[coin] = init_balance

    def get_money(self, coin, amount):

        try:
            coin_balance = self.wallet.loc[coin].Balance
        except KeyError:
            return 0

        if amount > coin_balance:
            # print("Insufficient balance: Available: %s " % coin_balance)
            return 0
        else:
            self.wallet.loc[coin] = coin_balance - amount
            # print("New balance = %s %s" % (self.wallet.loc[coin].Balance, coin))
            return self.wallet.loc[coin].Balance

    def put_money(self, coin, amount):

        try:
            coin_balance = self.wallet.loc[coin].Balance
        except KeyError:
            coin_balance = 0
            pass

        self.wallet.loc[coin] = coin_balance + amount

    def get_balance(self, coin='USDT'):

        if not coin:
            return self.wallet

        try:
            coin_balance = self.wallet.loc[coin].Balance
        except KeyError:
            return 0

        return coin_balance

    def get_balance_usdt(self, coins):

        usdt_balance = 0

        # coins = [{'coin': 'BTC', 'price': 0}]

        for coin, price in coins.items():
            try:
                usdt_balance += self.wallet.loc[coin].Balance * price
            except KeyError:
                usdt_balance += 0

        usdt_balance += self.wallet.loc['USDT'].Balance

        return usdt_balance


class Trade:

    newid = itertools.count()

    def __init__(self, open_price, position_size):

        self.id = next(Trade.newid)

        self.open_price = open_price
        self.position_size = 0
        self.position_avg_price = 0

        self.df_trades = pd.DataFrame(columns=['Date', 'Coin', 'Price', 'Size', 'Type'])

        self.status_open = True

        self.kill = False

    def __iter__(self):
        for each in self.__dict__.values():
            yield each

    def w_avg(self, df, values, weights):

        d = df[values]
        w = df[weights]
        return (d * w).sum() / w.sum()

    def position_value(self, actual_price):

        pos_value = self.df_trades['Size'].sum() * actual_price

        return pos_value

    def position_profit(self, actual_price):

        if actual_price < self.open_price:

            profit = ((actual_price - self.position_avg_price)/self.position_avg_price) * 100
        else:
            profit = ((actual_price - self.position_avg_price)/self.position_avg_price) * 100

        return profit

    def profit_value(self, actual_price):

        return (actual_price * self.df_trades['Size'].sum()) - (self.position_avg_price * self.df_trades['Size'].sum())

    def add_position(self, coin, date_time, price, size, ptype):

        self.position_size += size/price

        self.df_trades.loc[self.df_trades.shape[0]] = [date_time, coin, price, size/price, ptype]

        self.position_avg_price = self.avg_price()

        return size/price

    def avg_price(self):

        d = self.df_trades['Price']
        w = self.df_trades['Size']

        return (d * w).sum() / w.sum()

    def total_time(self):
        total = (self.df_trades.Date.iloc[-1] - self.df_trades.Date.iloc[0])
        return total



class SimpleDCABot:

    def __init__(self, condition, pair, bag, max_open_trades, ttp, base_order_size, safety_order_size, sos, max_safety_orders, safety_scale, safe_order_step_scale, debug=False):

        self.pair = pair
        self.logger = logger
        self.ttp = ttp
        self.base_order_size = base_order_size
        self.safety_order_size = safety_order_size
        self.sos = sos
        self.max_safety_orders = max_safety_orders
        self.safe_order_step_scale = safe_order_step_scale
        self.safety_scale = safety_scale
        self.max_open_trades = max_open_trades
        self.open_trades = 0
        self.current_so = 0
        self.balance = bag
        self.kill = False
        self.condition = condition
        self.debug = debug

    def _communicate(self, msg):

        if self.debug:
            logger.info(f'{msg}')

    def build_safety_orders(self, current_so, safe_order_volume_scale, safety_order_size):

        # return the size (volume) of the safety order

        return safety_order_size * (safe_order_volume_scale ** (current_so-1))

    def run(self, coin):
        logger.info(f'SimpleDCABot running...')

        th = threading.Thread(target=self.start)
        th.start()

        self._communicate('Checking API settings...')

    def start(self):

        coin_a = self.pair[0:3]
        coin_b = self.pair[3:7]

        test_data = pd.read_sql(self.pair, engine).set_index('Time')

        if self.condition in test_data.columns:
            condition = self.condition
        else:
            condition = 'Open'

        trade = []

        trade_index = 0

        total_opened_trades = 0
        total_closed_trades = 0

        item = None

        self._communicate("Start balance = %s " % self.balance.get_balance('USDT'))

        for i, item in enumerate(test_data.itertuples(), 1):
            '''
            if trade:
                print("%s - Preço: %s - Open Trades: %s - Profit: %s - Profit %s pct - Avg Price: %s - Total SO: %s" %
                      (item.Index.strftime("%m/%d/%Y, %H:%M:%S"), item.Close, self.open_trades,
                       round(trade[trade_index].position_profit(item.Close), 2) if self.open_trades else 0, price_deviation, round(trade[trade_index].avg_price(), 2) if self.open_trades else 0, self.current_so - 1))
            '''

            if getattr(item, condition) > 0 and self.open_trades < self.max_open_trades:

                self._communicate("Open trade at %s with price of: %s" % (item.Index, item.Close))

                if self.balance.get_money(coin_b, self.base_order_size):

                    trade.append(Trade(item.Close, self.base_order_size))

                    amount = trade[trade_index].add_position(coin_a, item.Index, item.Close, self.base_order_size, 'BO')

                    self.balance.put_money(coin_a, amount)

                    self.open_trades += 1

                    self.current_so += 1

                    total_opened_trades += 1

                else:
                    self._communicate("No balance to make trade")

            if self.open_trades > 0 and trade:
                # check profit/loss
                price_deviation = trade[trade_index].position_profit(item.Close)

                if price_deviation >= self.ttp:
                    # take profit, close trade
                    profit = trade[trade_index].profit_value(item.Close)

                    total_trade = trade[trade_index].position_value(item.Close)

                    self.balance.get_money(coin_a, trade[trade_index].position_value(1))

                    self.balance.put_money(coin_b, total_trade)

                    self._communicate("Closing trade at %s - profit = %.3f (%.3f %%) - New total balance = %.2f" % (item.Close, profit, price_deviation, self.balance.get_balance(coin_b)))

                    self.open_trades = 0
                    self.current_so = 0

                    trade[trade_index].status_open = False

                    trade_index += 1
                    total_closed_trades += 1

            if self.open_trades > 0 and trade:
                if self.safe_order_step_scale == 1:
                    threshold = trade[trade_index].open_price - (trade[trade_index].open_price * self.sos * self.safe_order_step_scale * self.current_so)/100
                else:
                    threshold = trade[trade_index].open_price - (trade[trade_index].open_price * (((self.safe_order_step_scale ** self.current_so) - 1)  / (self.safe_order_step_scale - 1)/100))

                if item.Close <= threshold and self.current_so < (self.max_safety_orders + 1):
                    # open safety trade

                    so_size = self.build_safety_orders(self.current_so, self.safety_scale, self.safety_order_size)

                    self._communicate("Open safety trade: %s / %s - SO size = %s" % (self.current_so, self.max_safety_orders, so_size))

                    if self.balance.get_money(coin_b, so_size):

                        amount = trade[trade_index].add_position(coin_a, item.Index, item.Close, so_size, 'SO_' + str(self.current_so))

                        self.balance.put_money(coin_a, amount)

                        self.current_so += 1
                    else:
                        self._communicate("Insufficient funds to make SO")
        if item:
            total_usdt = self.balance.get_balance_usdt({coin_a: item.Close})
        else:
            total_usdt = self.balance.get_balance('USDT')

        self._communicate("Final balance = Total %s %s (%s %s / %s %s) - Total trades = %s -  Total Closed trades = %s" % (coin_b, total_usdt, coin_b, self.balance.get_balance('USDT'), coin_a, self.balance.get_balance('BTC'), total_opened_trades, total_closed_trades))

        return trade, total_usdt, total_opened_trades, total_closed_trades

    def kill(self):

        self.kill = True


def getminutedata(symbol, lookback='360'):
    print("Loading Binance data for %s" % symbol)
    frame = pd.DataFrame(client.get_historical_klines(symbol, '5m', lookback + ' days ago UTC'))
    print("Loading OK")

    frame = frame.iloc[:, :5]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close']
    frame[['Open', 'High', 'Low', 'Close']] = frame[['Open', 'High', 'Low', 'Close']].astype(float)
    frame.Time = pd.to_datetime(frame.Time, unit='ms')

    return frame


class Signals:

    def __init__(self, df, lags):
        self.df = df
        self.lags = lags

    def gettrigger(self):

        dfx = pd.DataFrame()

        for i in range(self.lags + 1):
            mask = (self.df['%K'].shift(i) < 20) & (self.df['%D'].shift(i) < 20)
            dfx = dfx.append(mask, ignore_index=True)

        return dfx.sum(axis=0)

    def decide(self):

        self.df['trigger'] = np.where(self.gettrigger(), 1, 0)
        self.df['Buy'] = np.where((self.df.trigger) & (self.df['%K'].between(20, 80)) & (self.df['%D'].between(20, 80)) & (self.df.rsi > 50) & (self.df.macd > 0), 1, 0)


def technicals(df, ema_slow1=200, ema_fast1=50, sma_slow2=25, sma_fast2=7):

    # df = df.copy()

    # calculates the percentage change between the current and a prior element

    # df['return'] = np.log(df.Close.pct_change() + 1)

    #df['SMA_fast'] = df.Close.rolling(7).mean()
    #df['SMA_slow'] = df.Close.rolling(25).mean()

    df['%K'] = ta.momentum.stoch(df.High, df.Low, df.Close, window=14, smooth_window=3)
    df['%D'] = df['%K'].rolling(3).mean()
    df['rsi'] = ta.momentum.rsi(df.Close, window=14)
    df['macd'] = ta.trend.macd_diff(df.Close)

    df['EMA200'] = ta.trend.ema_indicator(df.Close, window=ema_slow1)

    df['SMA50'] = ta.trend.sma_indicator(df.Close, window=20)
    df['SMA20'] = ta.trend.sma_indicator(df.Close, window=50)
    df['SMA100'] = ta.trend.sma_indicator(df.Close, window=100)

    df['EMA_slow1'] = ta.trend.ema_indicator(df.Close, window=ema_slow1)
    df['EMA_fast1'] = ta.trend.ema_indicator(df.Close, window=ema_fast1)

    df['SMA_fast2'] = ta.trend.sma_indicator(df.Close, window=sma_fast2)
    df['SMA_slow2'] = ta.trend.sma_indicator(df.Close, window=sma_slow2)

    df['Buy_sma2'] = 0.0
    df['Buy_sma2'] = np.where((df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA100']), 1.0, 0.0)

    df['position_sma'] = df['Buy_sma2'].diff()

    df['wf_Top_bool'] = np.where(df['High'] == df['High'].rolling(9, center=True).max(), True, False)

    df['wf_Top'] = np.where(df['High'] == df['High'].rolling(9, center=True).max(), df['High'], np.NaN)

    df['wf_Top'] = df['wf_Top'].ffill()

    df.dropna(inplace=True)

    # william's fractal indicator strategy

    df['Buy_fractal'] = np.where((df.Close > df.wf_Top) & (df.Close > df.EMA200), 1, 0)

    df['SL'] = np.where(df.Buy_fractal == 1, df.Close - (df.Close - df.Low), 0)
    df['TP'] = np.where(df.Buy_fractal == 1, df.Close + (df.Close - df.Low) * 1.5, 0)

    df.dropna(inplace=True)

    # df['position'] = np.where(df['SMA_fast'] > df['SMA_slow'], 1, 0)
    # df['strategyreturn'] = df['position'].shift(1) * df['return']

    # df.Time = pd.to_datetime(df.Time, unit='ms')
    # df['Time2'] = pd.to_datetime(df.Time, unit='ms')
    # df.dropna(inplace=True)

    # test = Signals(df, 3)

    # test.decide()

    return df


def get_high_tf(df):
    df = df.set_index('Time')
    # keep a column with date time
    df["Time"] = df.index

    agg_dict = {'Time': 'last',
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Adj Close': 'last',
                'Volume': 'mean'}

    agg_dict = { 'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'}

    # create 4hour dataframe
    df_4hour2 = df.resample('4h').agg(agg_dict)

    # keep a column with date time
    df_4hour2["Time"] = df_4hour2.index

    # merge with the lower time frame

    # pd.merge(df, df_4hour2[['Close']], how='left', left_index=True, right_index=True)

    df_day = df.resample('1D').last()

    df_day["Time"] = df_day.index

    merged = pd.merge(df, df_4hour2[['Close']], how='left', left_index=True, right_index=True)

    merged = merged.rename({'Close_y': 'Close_4H', 'Close_x': 'Close'}, axis=1)

    merged['Close_4H'] = merged['Close_4H'].ffill()

    merged = pd.merge(merged, df_day[['Close']], how='left', left_index=True, right_index=True)

    merged = merged.rename({'Close_y': 'Close_1D', 'Close_x': 'Close'}, axis=1)

    merged['Close_1D'] = merged['Close_1D'].ffill()

    df = merged.dropna()

    df['1D_SMA20'] = ta.trend.sma_indicator(df.Close_1D, window=20)
    df['1D_SMA50'] = ta.trend.sma_indicator(df.Close_1D, window=50)

    df['4H_SMA20'] = ta.trend.sma_indicator(df.Close_4H, window=20)
    df['4H_SMA50'] = ta.trend.sma_indicator(df.Close_4H, window=50)

    return df


def plot_chart(df):

    fig = plt.figure(figsize=(12, 10))

    price_ax = plt.subplot(2, 1, 1)

    # plt.figure(figsize=(12, 6))

    price_ax.plot(df[['Close', 'SMA20', 'SMA50']])

    price_ax.plot(df[df.position_sma == 1].index, df['Close'][df['position_sma'] == 1], linestyle='None', marker='^',
                  markersize=15, color='g', label='buy')

    # (df['Buy'] * df['Close']).plot(label='Buy', kind='bar')
    price_ax.legend(['Close', 'SMA20', 'SMA50', 'SMA100'], loc="upper left")

    roc_ax = plt.subplot(2, 1, 2, sharex=price_ax)

    roc_ax.plot(df[['rsi']], label="RSI", color="red")

    roc_ax.legend(loc="upper left")
    price_ax.set_title("BTC Prices and SMA/RSI indicators")

    # Removing the date labels and ticks from the price subplot:
    price_ax.get_xaxis().set_visible(False)

    # Removing the gap between the plots:
    fig.subplots_adjust(hspace=0)
    # Adding a horizontal line at the zero level in the ROC subplot:
    roc_ax.axhline(50, color=(.5, .5, .5), linestyle='--', alpha=0.5)
    roc_ax.axhline(20, color=(.5, .5, .5), linestyle='--', alpha=0.5)
    roc_ax.axhline(80, color=(.5, .5, .5), linestyle='--', alpha=0.5)

    # We can add labels to both vertical axis:
    price_ax.set_ylabel("Price ($)")
    roc_ax.set_ylabel("RSI")

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.show()


def apply_technicals(pair='BTCUSDT'):

    test_data = pd.read_sql(pair, engine).set_index('Time')

    df = technicals(test_data)

    print("Done")


def get_binance_data(coins, engine, days='10'):

    for coin in tqdm(coins):

        frame = getminutedata(coin, days)

        frame = technicals(frame)

        frame.to_sql(coin, engine, index=False)

    # test = pd.read_sql('BTCUSDT', engine).set_index('Time')
    # technicals(test)
    print("downloaded binance data - ok")


def safe_arange(start, stop, step):
    return (step * np.arange(start / step, stop / step)).tolist()


def createList(r1, r2, r3):
    return np.arange(r1, r2+r3, r3)


def test_bot(wallet_init, strategy, profit, bo, so, sos, qtd_so, saf_scale, saf_step_scale, debug=False):

    # pair, bag, max_open_trades,
    # ttp, base_order_size, safety_order_size, sos, max_safety_orders, volume_scale, safety_scale, safe_order_step_scale):

    a = [profit, bo, so, sos, qtd_so, saf_scale, saf_step_scale]

    a = list(itertools.product(*a))

    results = []

    for test in tqdm(a):
        dict1 = {}
        account = AccountBag('USDT', wallet_init)

        bot = SimpleDCABot(strategy, 'BTCUSDT', account, 1, *test, debug=debug)

        result, total_usdt, total_trades, total_closed = bot.start()

        dict1.update({'Test': test, 'Total_USDT': total_usdt, 'Final_USDT': account.get_balance("USDT"), 'Final_BTC': account.get_balance("BTC"), 'Total_opened_trades': total_trades, 'Total_closed_trades': total_closed, 'Bot_trades': result})

        results.append(dict1)

    df = pd.DataFrame(results).sort_values(by=['Total_USDT'], ascending=False)

    return df


if __name__ == '__main__':

    engine = sqlalchemy.create_engine('sqlite:///Cryptoprices.db')

    fname = 'Cryptoprices.db'

    coins = ('BTCUSDT',)

    days_str = '400'

    wallet_init = 10000

    # strategy = 'position_sma'
    strategy = 'always'

    profit = safe_arange(1, 3.5, 0.5)
    bo = safe_arange(100, 120, 20)
    so = safe_arange(100, 220, 20)
    sos = safe_arange(1, 3.5, 0.5)
    qtd_so = safe_arange(4, 8, 1)
    saf_scale = safe_arange(1, 2, 1)
    saf_step_scale = safe_arange(1, 1.3, 0.1)

    if not os.path.isfile(fname):
        get_binance_data(coins, engine, days_str)

    results = test_bot(wallet_init, strategy, profit, bo, so, sos, qtd_so, saf_scale, saf_step_scale, debug=True)

    print("Fim")






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
