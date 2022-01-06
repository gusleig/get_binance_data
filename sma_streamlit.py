import pandas as pd
import websocket, json
from sqlalchemy import create_engine

engine = create_engine('sqlite:///CryptoDB_live.db')

stream1 = 'wss://stream2.binance.com:9443/ws/!miniTicker@arr@3000ms'
stream2 = "wss://stream.binance.com:9443/ws/ltcbtc@aggTrade/ethbtc@aggTrade"
stream3 = 'wss://stream.binance.com:9443/stream?streams=ethbtc@kline_5m'
stream4 = "wss://stream.binance.com:9443/stream?streams=ethbtc@ticker"
stream5 = "wss://stream.binance.com:9443/stream?streams=!ticker@arr"
stream6 = "wss://stream.binance.com:9443/stream?streams=ethbtc@depth5"
stream7 = "wss://stream.binance.com:9443/stream?streams=ethbtc@depth"

stream = "wss://stream2.binance.com:9443/ws/!miniTicker@arr"


def on_open(ws):
    print('opened connection')


def on_close(ws):
    print('closed connection')


def on_error(ws, error):
    print(error)


def on_message(ws, message):
    msg = json.loads(message)

    # filter usdt pairs
    symbol = [x for x in msg if x['s'].endswith('USDT')]
    frame = pd.DataFrame(symbol)[['E', 's', 'c']]
    frame.E = pd.to_datetime(frame.E, unit='ms')
    frame.c = frame.c.astype(float)

    for row in range(len(frame)):
        data = frame[row:row+1]
        print(data['E'].values[0])
        data[['E', 'c']].to_sql(data['s'].values[0], engine, index=False, if_exists='append')


if __name__ == "__main__":

    ws = websocket.WebSocketApp(stream, on_error=on_error, on_open=on_open, on_close=on_close, on_message=on_message)
    ws.run_forever()

