import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Crypto Quant Terminal", layout="wide", page_icon="📈")

# --- FUNZIONI BACKEND ---
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    # Lista di API da provare (Global e US per evitare blocchi geografici)
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}", # Global
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"   # US Backup
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status() # Controlla se c'è errore 403/404
            data = response.json()
            
            # Se abbiamo dati, procediamo
            if len(data) > 0:
                df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
                cols = ['open', 'high', 'low', 'close', 'vol']
                df[cols] = df[cols].astype(float)
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                return df
        except Exception:
            continue # Se fallisce, prova il prossimo URL
            
    st.error(f"❌ Impossibile connettersi a Binance per {symbol}. Verifica il simbolo.")
    return pd.DataFrame()

def calculate_indicators(df):
    if df.empty: return df
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # SMA & EMA
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df

def generate_signal(row):
    trend = "BULLISH" if row['close'] > row['ema50'] else "BEARISH"
    rsi_status = "NEUTRAL"
    if row['rsi'] > 70: rsi_status = "OVERBOUGHT"
    if row['rsi'] < 30: rsi_status = "OVERSOLD"
    
    signal = "NEUTRAL"
    if trend == "BULLISH" and 40 < row['rsi'] < 60:
        signal = "LONG ENTRY"
    elif trend == "BEARISH" and 40 < row['rsi'] < 60:
        signal = "SHORT ENTRY"
    elif rsi_status == "OVERSOLD" and trend == "BULLISH":
        signal = "STRONG BUY (Dip)"
        
    return signal, trend, rsi_status

# --- UI ---
st.title("⚡ Crypto Quant Terminal")

# Sidebar
st.sidebar.header("Parametri")
symbol_input = st.sidebar.text_input("Simbolo", value="BTCUSDT").upper()
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

# Logica di caricamento automatico o manuale
if st.button("ANALIZZA MERCATO", type="primary") or "data_loaded" not in st.session_state:
    with st.spinner('Connessione ai server Binance...'):
        df = get_binance_data(symbol_input, interval)
        
        if not df.empty:
            df = calculate_indicators(df)
            st.session_state['data'] = df
            st.session_state['data_loaded'] = True
        else:
            st.warning("Nessun dato ricevuto. Riprova tra poco.")

# Visualizzazione (Se i dati esistono)
if 'data' in st.session_state and not st.session_state['data'].empty:
    df = st.session_state['data']
    curr = df.iloc[-1]
    signal, trend, rsi_status = generate_signal(curr)
    
    # KPI
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Prezzo", f"${curr['close']:,.2f}")
    kpi2.metric("Segnale", signal, delta_color="normal" if signal=="NEUTRAL" else "inverse")
    kpi3.metric("RSI (14)", f"{curr['rsi']:.1f}", rsi_status)
    kpi4.metric("ATR (Volatilità)", f"${curr['atr']:.2f}")
    
    # Trading Plan
    st.markdown("### 📋 Piano Operativo")
    c1, c2, c3 = st.columns(3)
    atr_sl = curr['atr'] * 2.0
    
    if "LONG" in signal or "BUY" in signal:
        c1.success(f"**ENTRY:** ${curr['close']:,.2f}")
        c2.error(f"**STOP LOSS:** ${curr['close'] - atr_sl:,.2f}")
        c3.info(f"**TAKE PROFIT:** ${curr['close'] + atr_sl:,.2f}")
    elif "SHORT" in signal:
        c1.error(f"**ENTRY:** ${curr['close']:,.2f}")
        c2.error(f"**STOP LOSS:** ${curr['close'] + atr_sl:,.2f}")
        c3.success(f"**TAKE PROFIT:** ${curr['close'] - atr_sl:,.2f}")
    else:
        st.info("Nessun segnale operativo chiaro al momento.")

    # Grafico
    fig = go.Figure(data=[go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')])
    fig.add_trace(go.Scatter(x=df['ts'], y=df['ema50'], line=dict(color='blue', width=1), name='EMA 50'))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
