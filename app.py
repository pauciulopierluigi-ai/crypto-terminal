import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Crypto Quant Terminal", layout="wide", page_icon="📈")

# --- FUNZIONI BACKEND ---
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=5)
        data = response.json()
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
        
        # Conversione tipi
        cols = ['open', 'high', 'low', 'close', 'vol']
        df[cols] = df[cols].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Errore API Binance: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # SMA & EMA
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df

def generate_signal(row):
    # Logica Quantitativa
    trend = "BULLISH" if row['close'] > row['ema50'] else "BEARISH"
    rsi_status = "NEUTRAL"
    if row['rsi'] > 70: rsi_status = "OVERBOUGHT"
    if row['rsi'] < 30: rsi_status = "OVERSOLD"
    
    signal = "NEUTRAL"
    
    # Condizioni Operative
    if trend == "BULLISH" and 40 < row['rsi'] < 60:
        signal = "LONG ENTRY"
    elif trend == "BEARISH" and 40 < row['rsi'] < 60:
        signal = "SHORT ENTRY"
    elif rsi_status == "OVERSOLD" and trend == "BULLISH":
        signal = "STRONG BUY (Dip)"
        
    return signal, trend, rsi_status

# --- INTERFACCIA UTENTE (UI) ---
st.title("⚡ Crypto Quant Terminal")
st.markdown("Analisi istituzionale in tempo reale • *Powered by Python*")

# Sidebar
st.sidebar.header("Parametri")
symbol_input = st.sidebar.text_input("Simbolo (es. BTCUSDT)", value="BTCUSDT").upper()
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

if st.button("ANALIZZA MERCATO") or st.sidebar.button("Aggiorna"):
    with st.spinner('Scaricamento dati e calcolo algo...'):
        df = get_binance_data(symbol_input, interval)
        
        if not df.empty:
            df = calculate_indicators(df)
            curr = df.iloc[-1]
            
            signal, trend, rsi_status = generate_signal(curr)
            
            # Calcolo SL/TP dinamici
            atr_sl = curr['atr'] * 2.0
            stop_loss = curr['close'] - atr_sl if "LONG" in signal or "BUY" in signal else curr['close'] + atr_sl
            take_profit = curr['close'] + atr_sl if "LONG" in signal or "BUY" in signal else curr['close'] - atr_sl
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Prezzo Attuale", f"${curr['close']:,.2f}", f"{curr['close'] - df.iloc[-2]['close']:.2f}")
            col2.metric("Segnale Algo", signal, delta_color="normal" if signal=="NEUTRAL" else "inverse")
            col3.metric("RSI (14)", f"{curr['rsi']:.1f}", rsi_status)
            col4.metric("Volatilità (ATR)", f"${curr['atr']:.2f}")
            
            # Sezione Trading Plan
            st.markdown("### 📋 Trading Plan Generato")
            plan_col1, plan_col2, plan_col3 = st.columns(3)
            
            if signal != "NEUTRAL":
                plan_col1.info(f"**ENTRY:** ${curr['close']:,.2f}")
                plan_col2.error(f"**STOP LOSS:** ${stop_loss:,.2f}")
                plan_col3.success(f"**TAKE PROFIT:** ${take_profit:,.2f}")
            else:
                st.warning("⚠️ Mercato in conflitto o laterale. Attendere configurazione migliore.")

            # Grafico Interattivo
            st.markdown("---")
            fig = go.Figure(data=[go.Candlestick(x=df['ts'],
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='Price')])
            
            fig.add_trace(go.Scatter(x=df['ts'], y=df['sma20'], line=dict(color='orange', width=1), name='SMA 20'))
            fig.add_trace(go.Scatter(x=df['ts'], y=df['ema50'], line=dict(color='blue', width=1), name='EMA 50'))
            
            fig.update_layout(title=f"{symbol_input} Chart ({interval})", xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw Data
            with st.expander("Vedi Dati Grezzi"):
                st.dataframe(df.tail(10))
