import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Pro Quant Terminal", layout="wide", page_icon="💎")

# CSS Custom per look professionale
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stAlert {margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# --- LISTA TOP 20 CRYPTO (Aggiornata staticamente per velocità) ---
TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    "TONUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", 
    "NEARUSDT", "APTUSDT"
]

# --- FUNZIONI BACKEND (Caching per performance) ---
@st.cache_data(ttl=60) # Cache valida per 60 secondi
def get_market_data(symbol, interval, limit=150):
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    ]
    
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            if not data: continue
            
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
            cols = ['open', 'high', 'low', 'close', 'vol']
            df[cols] = df[cols].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except Exception:
            continue
    return pd.DataFrame()

def calculate_advanced_indicators(df):
    if df.empty: return df
    
    # 1. Trend
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Volatilità (ATR & Bollinger)
    df['TR'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift()), 
                          abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (std * 2)
    df['BB_Lower'] = df['SMA20'] - (std * 2)
    
    return df

def strategy_engine(df, macro_sentiment, nasdaq_corr, selling_wall_pct):
    curr = df.iloc[-1]
    
    # --- 1. ANALISI TECNICA BASE ---
    trend = "BULLISH" if curr['close'] > curr['EMA50'] else "BEARISH"
    macd_cross = "BULLISH" if curr['MACD'] > curr['Signal'] else "BEARISH"
    rsi_state = "NEUTRAL"
    if curr['RSI'] > 70: rsi_state = "OVERBOUGHT"
    if curr['RSI'] < 30: rsi_state = "OVERSOLD"
    
    # --- 2. CALCOLO SCORE (0-10) ---
    score = 5
    if trend == "BULLISH": score += 2
    if macd_cross == "BULLISH": score += 1
    if 40 < curr['RSI'] < 60: score += 1 # Zona momentum sana
    if curr['close'] > curr['SMA20']: score += 1
    
    # Penalità Macro (Dinamica)
    atr_multiplier_sl = 2.0
    
    if macro_sentiment == "Negativo (Risk Off)":
        score -= 2
        atr_multiplier_sl = 3.0 # Allarga SL per volatilità
    elif macro_sentiment == "Positivo (Risk On)":
        score += 1
        atr_multiplier_sl = 1.5 # Stringi SL
        
    if nasdaq_corr == "Divergenza Ribassista":
        score -= 1.5
        
    # --- 3. PIANO ESECUTIVO DINAMICO ---
    stop_loss = 0
    tp1 = 0
    tp2 = 0
    action = "WAIT"
    
    # Logica Long
    if score >= 6.5:
        action = "LONG ENTRY"
        stop_loss = curr['close'] - (curr['ATR'] * atr_multiplier_sl)
        
        # TP dinamico in base al Muro di Vendita
        wall_price = curr['close'] * (1 + (selling_wall_pct/100))
        tp1 = min(curr['close'] + (curr['ATR'] * 2), wall_price * 0.995) # Front-run the wall
        tp2 = curr['close'] + (curr['ATR'] * 4)
        
    # Logica Short
    elif score <= 3.5:
        action = "SHORT ENTRY"
        stop_loss = curr['close'] + (curr['ATR'] * atr_multiplier_sl)
        tp1 = curr['close'] - (curr['ATR'] * 2)
        tp2 = curr['close'] - (curr['ATR'] * 4)
        
    return {
        "action": action,
        "score": min(max(score, 0), 10),
        "sl": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": curr['RSI'],
        "atr": curr['ATR'],
        "trend": trend,
        "macd": macd_cross
    }

# --- INTERFACCIA LATERALE (INPUT) ---
st.sidebar.title("⚙️ Control Room")

# 1. Selezione Asset
selected_symbol = st.sidebar.selectbox("Crypto Asset", TOP_COINS, index=0)
timeframe = st.sidebar.select_slider("Timeframe", options=["15m", "1h", "4h", "1d"], value="1h")

st.sidebar.markdown("---")
st.sidebar.header("🌍 Fattori Esterni (Macro)")
st.sidebar.info("Modifica questi valori in base alle news per adattare la strategia in tempo reale.")

# 2. Input Macro
macro_status = st.sidebar.radio("Sentiment Macro / News (FED, CPI)", 
                 ["Neutrale", "Positivo (Risk On)", "Negativo (Risk Off)"])

nasdaq_input = st.sidebar.selectbox("Correlazione Nasdaq/SP500", 
                 ["Normale", "Divergenza Ribassista (Tech Crolla)", "Divergenza Rialzista"])

# 3. Input Order Book
wall_input = st.sidebar.slider("Muro di Vendita (Order Book)", 
                 min_value=0.0, max_value=10.0, value=2.0, step=0.5, format="+%f%%")

if st.sidebar.button("AGGIORNA ANALISI", type="primary"):
    st.session_state['refresh'] = True

# --- MAIN PAGE ---
st.title(f"📊 Analisi Algoritmica: {selected_symbol}")

# Caricamento Dati
with st.spinner('Scaricamento dati e calcolo modelli quantitativi...'):
    df = get_market_data(selected_symbol, timeframe)
    
    if not df.empty:
        df = calculate_advanced_indicators(df)
        
        # Esecuzione Strategia
        strat = strategy_engine(df, macro_status, nasdaq_input, wall_input)
        curr_price = df['close'].iloc[-1]
        price_change = curr_price - df['open'].iloc[-1]
        pct_change = (price_change / df['open'].iloc[-1]) * 100
        
        # --- SEZIONE 1: KPI CARDS ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Prezzo Attuale", f"${curr_price:,.2f}", f"{pct_change:.2f}%")
        k2.metric("Confidence Score", f"{strat['score']}/10", delta_color="off")
        k3.metric("RSI (Momentum)", f"{strat['rsi']:.1f}", strat['action'])
        k4.metric("Volatilità (ATR)", f"${strat['atr']:.2f}", f"Stop Buffer: {strat['atr']*2:.0f}")
        
        # --- SEZIONE 2: PIANO OPERATIVO (Actionable) ---
        st.markdown("### 🎯 Piano Esecutivo Dinamico")
        
        # Container colorato in base all'azione
        container_color = "gray"
        if "LONG" in strat['action']: container_color = "rgba(40, 167, 69, 0.2)"
        elif "SHORT" in strat['action']: container_color = "rgba(220, 53, 69, 0.2)"
        
        with st.container():
            st.markdown(f"""
            <div style="background-color: {container_color}; padding: 20px; border-radius: 10px; border: 1px solid #555;">
                <h2 style="margin:0; text-align:center;">{strat['action']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if strat['action'] != "WAIT":
                col_sl, col_tp1, col_tp2 = st.columns(3)
                col_sl.error(f"🛑 STOP LOSS: ${strat['sl']:,.2f}")
                col_tp1.warning(f"💰 TAKE PROFIT 1: ${strat['tp1']:,.2f}")
                col_tp2.success(f"🚀 TAKE PROFIT 2: ${strat['tp2']:,.2f}")
                
                # Risk Metrics
                risk = abs(curr_price - strat['sl'])
                reward = abs(strat['tp1'] - curr_price)
                rr_ratio = reward / risk if risk > 0 else 0
                st.caption(f"📊 Rischio stimato per coin: ${risk:.2f} | R/R Ratio al TP1: 1:{rr_ratio:.2f}")

        # --- SEZIONE 3: GRAFICO INTERATTIVO ---
        st.markdown("---")
        fig = go.Figure()
        
        # Candele
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name='Prezzo'))
        
        # Indicatori
        fig.add_trace(go.Scatter(x=df['ts'], y=df['SMA20'], line=dict(color='orange', width=1), name='SMA 20'))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Boll Upper'))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Boll Lower'))
        
        # Livelli Operativi (Se attivi)
        if strat['action'] != "WAIT":
            fig.add_hline(y=strat['sl'], line_dash="dash", line_color="red", annotation_text="SL")
            fig.add_hline(y=strat['tp1'], line_dash="dash", line_color="green", annotation_text="TP1")
            
        fig.update_layout(title=f"{selected_symbol} Technical Chart ({timeframe})", 
                          height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- SEZIONE 4: TABELLA DATI AVANZATA ---
        st.markdown("### 📋 Analisi Dettagliata (Ultimi 5 Periodi)")
        
        # Creiamo un dataframe più leggibile per l'utente
        view_df = df[['ts', 'close', 'RSI', 'MACD', 'ATR', 'vol']].tail(5).copy()
        view_df['ts'] = view_df['ts'].dt.strftime('%Y-%m-%d %H:%M')
        view_df = view_df.sort_values(by='ts', ascending=False)
        
        st.dataframe(view_df.style.background_gradient(subset=['RSI'], cmap='coolwarm'), use_container_width=True)

    else:
        st.error("Errore nel recupero dei dati. Riprova o cambia asset.")
