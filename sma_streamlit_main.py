import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np

from sqlalchemy import create_engine

engine = create_engine('sqlite:///CryptoDB_live.db')

symbols = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"', engine).name.to_list()

st.title("Welcome to the live TA platform")


def applytechnicals(df):

    df['SMA_7'] = df.c.rolling(7).mean()
    df['SMA_25'] = df.c.rolling(25).mean()
    df.dropna(inplace=True)


def qry(symbol):

    now = dt.datetime.utcnow()
    before = now - dt.timedelta(minutes=30)
    qry_str = f"""SELECT E, c FROM '{symbol}' WHERE E >= '{before}'"""
    df = pd.read_sql(qry_str, engine)
    df.E = pd.to_datetime(df.E)
    df = df.set_index('E')
    df = df.resample('1min').last()
    applytechnicals(df)
    df['position'] = np.where(df['SMA_7'] > df['SMA_25'], 1, 0)

    return df


def check():
    for symbol in symbols:
        if len(qry(symbol).position) > 1:
            if qry(symbol).position[-1] and qry(symbol).position.diff()[-1]:
                st.write(symbol)


st.button('Get live SMA cross', on_click=check())
