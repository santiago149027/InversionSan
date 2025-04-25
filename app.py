import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Cargar modelo multiclase
best_model_multiclase = joblib.load("modelo_nvda_multiclase.pkl")

# Tickers y sectores asignados
acciones = {
    "NVDA": "Tecnología",
    "AAPL": "Tecnología",
    "MSFT": "Tecnología",
    "AMZN": "Consumo",
    "META": "Tecnología",
    "GOOGL": "Tecnología",
    "TSLA": "Consumo / Industrial",
    "ASML": "Tecnología",
    "VST": "Energía / Utilities",
    "TSM": "Tecnología",
    "NFLX": "Consumo / Media",
    "PLTR": "Tecnología",
    "GEV": "Energía",
    "HIMS": "Salud",
    "HOOD": "Financiero / Tecnología",
    "TEM": "Tecnología",
    "JPM": "Financiero",
    "LLY": "Salud",
    "AVGO": "Tecnología",
    "COIN": "Financiero / Cripto",
    "COST": "Consumo",
    "CRM": "Tecnología",
    "CSCO": "Tecnología",
    "DIS": "Consumo / Media",
    "ROST": "Consumo",
    "T": "Telecomunicaciones",
    "V": "Financiero",
    "MA": "Financiero",
    "SHOP": "Consumo",
    "WMT": "Consumo",
    "BITB": "Cripto",
    "MELI": "Consumo",
    "BABA": "Consumo / Tecnología",
    "PYPL": "Financiero / Tecnología",
    "CMG": "Consumo",
    "AMAT": "Tecnología",
    "CMCSA": "Telecomunicaciones",
    "A": "Tecnología",
    "FBTC": "Cripto",
    "BLK": "Financiero",
    "BRK-B": "Financiero",
    "CTSH": "Tecnología",
    "EPAM": "Tecnología",
    "IXN": "ETF Tecnología",
    "LMT": "Industrial / Defensa",
    "MRNA": "Salud",
    "HACK": "ETF Ciberseguridad",
    "ROBO": "ETF Robótica",
    "VCR": "ETF Consumo Discrecional",
    "VHT": "ETF Salud"
}

# Features requeridos por el modelo
features = [
    "RSI", "MACD", "MACD_Signal", "SMA_10", "EMA_10", "Momentum", "Volume",
    "bb_bbm", "bb_bbh", "bb_bbl", "bb_bandwidth",
    "atr", "cci", "adx", "roc"
]

# Función para preparar datos
def preparar_datos(ticker):
    df = yf.download(ticker, start="2018-01-01", end=datetime.today().strftime('%Y-%m-%d'), group_by='column')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or "Close" not in df.columns:
        return None, None, None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["RSI"] = RSIIndicator(close=close).rsi()
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["EMA_10"] = EMAIndicator(close=close, window=10).ema_indicator()
    df["Momentum"] = close.diff(4)
    bb = BollingerBands(close=close)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["bb_bandwidth"] = df["bb_bbh"] - df["bb_bbl"]
    atr = AverageTrueRange(high=high, low=low, close=close)
    df["atr"] = atr.average_true_range()
    cci = CCIIndicator(high=high, low=low, close=close)
    df["cci"] = cci.cci()
    adx = ADXIndicator(high=high, low=low, close=close)
    df["adx"] = adx.adx()
    roc = ROCIndicator(close=close)
    df["roc"] = roc.roc()

    df.columns = [str(c).strip() for c in df.columns]

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features].fillna(0)

    precio_actual = round(close.iloc[-1], 2)
    if len(close) > 1:
        variacion = round(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100, 2)
    else:
        variacion = None

    return df, precio_actual, variacion

# Configurar Streamlit
st.set_page_config(page_title="Dashboard de Señales de Trading", layout="wide")
st.title("📈 Dashboard de Recomendaciones de Trading")
st.caption("Modelo basado en NVDA: 📈 Comprar / ➖ Mantener / 🔻 Vender")

# Procesar acciones
resultados = []

for ticker, sector in acciones.items():
    df, precio, variacion = preparar_datos(ticker)

    if df is None or len(df) < 30:
        resultados.append({
            "Ticker": ticker,
            "Sector": sector,
            "Precio actual": "N/A",
            "Variación (%)": "N/A",
            "Recomendación": "❌ Datos insuficientes"
        })
        continue

    try:
        latest_data = df.tail(1)
        pred = best_model_multiclase.predict(latest_data)[0]
        if pred == 2:
            señal = "📈 Comprar"
        elif pred == 1:
            señal = "➖ Mantener"
        else:
            señal = "🔻 Vender"

        resultados.append({
            "Ticker": ticker,
            "Sector": sector,
            "Precio actual": f"${precio}",
            "Variación (%)": f"{variacion}%",
            "Recomendación": señal
        })
    except Exception as e:
        resultados.append({
            "Ticker": ticker,
            "Sector": sector,
            "Precio actual": "N/A",
            "Variación (%)": "N/A",
            "Recomendación": f"⚠️ Error: {str(e)}"
        })

# Mostrar resultados por sector (CORREGIDO)
df_resultados = pd.DataFrame(resultados)
sectores = df_resultados["Sector"].unique()

for sector in sorted(sectores):
    st.subheader(f"📂 Sector: {sector}")
    st.dataframe(
        df_resultados[df_resultados["Sector"] == sector]
        .set_index("Ticker")[["Precio actual", "Variación (%)", "Recomendación"]]
    )

# Explicación
st.markdown("""
---
🧠 **¿Cómo funciona este modelo?**  
Este sistema de IA analiza 15 indicadores técnicos y evalúa la probabilidad de que una acción suba o baje en los próximos 5 días.

- 📈 **Comprar** → Si se espera que el precio suba más de +1%
- 🔻 **Vender** → Si se espera que baje más de -1%
- ➖ **Mantener** → Si se espera que fluctúe dentro de ±1%
""")
