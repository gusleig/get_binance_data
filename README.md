# Binance DCA Bot backtesting
I needed a tool to test trading bot best configuration with historical data.

What is the best number of safety orders? What is the best threshold ?

To answer these questions you need to backtest your bot for each coin you are trading.

Loading historical data from Binance, The bot tries to mimit the 3commas bot using:

- profit. Profit percentage %.
- bo = Base order value.
- so = First Safety order value.
- sos = Percentage threshold to open a safety order
- qtd_so = Max number of safety order.
- saf_scale = Safety order volume scale.
- saf_step_scale = threshold multiplier for safety orders

You can specify the range you want for each parameter and the bot will test each combination.

With the value you put in the wallet parameter, at the end you can tell which configuration made more money.

## Strategy

To open the first trade the bot can use two strategies (strategy parameter):

- [always] Open as soon as possible, always open a new trade (total active trades = 1)
- [position_sma] Use SMA 20 > SMA 50 and SMA 50 > SMA 100 

The SMA strategy is described in the tecnincals function:

```
df['Buy_sma2'] = np.where((df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA100']), 1.0, 0.0)

df['position_sma'] = df['Buy_sma2'].diff()
```

## Configuration

create a config file config.ini with api_key and api_secret values
```
[API_CONFIG]
api_key = xxxx
api_secret = xxxx
```
or

use environmental variable with a .env file

```
api_key='xx'
api_secret='xx'
```

## Bot config

Open the main.py file search for the parameters. By now you can only use USDT pairs.

You can specify the range you want for each parameter and the bot will test each combination.

The bot will run with all the combinations and at the end you can check which one gets the higher balance in the wallet.

The function safe_arange gets the base value, top value and the step.

For instance, the profit would be tested from 1 to 3.5 with the parameter.

profit = safe_arange(1, 3.6, 0.5)

The days_str is the number of days behind today to get data from Binance. Is a string because the Binance API uses that way.

```
coins = ('BTCUSDT',)

days_str = '400'

wallet_init = 10000

profit = safe_arange(1, 3.6, 0.5)
bo = safe_arange(100, 120, 20)
so = safe_arange(100, 220, 20)
sos = safe_arange(1, 3.5, 0.5)
qtd_so = safe_arange(4, 8, 1)
vol_scale = safe_arange(1, 2, 1)
saf_scale = safe_arange(1, 2, 1)
saf_step_scale = safe_arange(1, 1.3, 0.1)

```
