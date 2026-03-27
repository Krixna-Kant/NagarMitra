import os
import pandas as pd
import numpy as np
import requests_cache
import openmeteo_requests
from retry_requests import retry
from datetime import datetime
import warnings
import joblib

# ML imports
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── STATION CONFIG ──
FETCH_STATIONS = ['Anand_Vihar', 'ITO', 'Pusa', 'Rohini', 'Okhla', 'Wazirpur']
DELHI_STATIONS = {
    'Anand_Vihar': {'lat': 28.6469, 'lon': 77.3152},
    'ITO':         {'lat': 28.6289, 'lon': 77.2400},
    'Pusa':        {'lat': 28.6358, 'lon': 77.1483},
    'Rohini':      {'lat': 28.7417, 'lon': 77.1020},
    'Okhla':       {'lat': 28.5380, 'lon': 77.2710},
    'Wazirpur':    {'lat': 28.7100, 'lon': 77.1682},
}

# ══════════════════════════════════════════════════════════════
# 1. FETCH & GENERATE DATA
# ══════════════════════════════════════════════════════════════
def fetch_weather():
    print("Fetching Weather Data (Open-Meteo)...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = 'https://archive-api.open-meteo.com/v1/archive'
    
    frames = []
    for name in FETCH_STATIONS:
        info = DELHI_STATIONS[name]
        params = {
            'latitude': info['lat'], 'longitude': info['lon'],
            'start_date': '2023-01-01', 'end_date': '2024-12-31',
            'hourly': ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                       'wind_direction_10m', 'precipitation', 'surface_pressure', 'boundary_layer_height'],
            'wind_speed_unit': 'kmh', 'timezone': 'Asia/Kolkata'
        }
        try:
            r = openmeteo.weather_api(url, params=params)[0]
            hourly = r.Hourly()
            time_range = pd.date_range(
                start=pd.Timestamp(hourly.Time(), unit='s', tz='Asia/Kolkata'),
                end=pd.Timestamp(hourly.TimeEnd(), unit='s', tz='Asia/Kolkata'),
                freq=pd.Timedelta(seconds=hourly.Interval()), inclusive='left'
            )
            df = pd.DataFrame({
                'datetime': time_range, 'station': name,
                'temperature': hourly.Variables(0).ValuesAsNumpy(),
                'humidity': hourly.Variables(1).ValuesAsNumpy(),
                'wind_speed': hourly.Variables(2).ValuesAsNumpy(),
                'wind_direction': hourly.Variables(3).ValuesAsNumpy(),
                'surface_pressure': hourly.Variables(5).ValuesAsNumpy(),
                'boundary_layer_height': hourly.Variables(6).ValuesAsNumpy(),
            })
            frames.append(df)
        except Exception as e:
            print(f"Failed {name}: {e}")
    return pd.concat(frames, ignore_index=True)

def generate_aqi(weather_df):
    print("Generating Realistic Synthetic AQI...")
    frames = []
    for i, name in enumerate(FETCH_STATIONS):
        np.random.seed(42 + i*7)
        df = weather_df[weather_df['station'] == name].copy().reset_index(drop=True)
        n = len(df)
        dt = df['datetime']
        month, hour, dow = dt.dt.month.values, dt.dt.hour.values, dt.dt.dayofweek.values
        
        # Base rules identical to original notebook
        sea = np.where((month>=11)|(month<=1), 280, np.where(month==10, 220, np.where(month==2, 200, np.where((month>=3)&(month<=5), 140, np.where((month>=6)&(month<=9), 75, 160)))))
        diu = np.where((hour>=8)&(hour<=10), 1.25, np.where((hour>=18)&(hour<=21), 1.20, np.where((hour>=22)|(hour<=5), 1.15, 0.90)))
        wk  = np.where(dow>=5, 0.85, 1.0)
        ws, blh = df['wind_speed'].fillna(10).values, df['boundary_layer_height'].fillna(800).values
        trp = np.where((ws<5)&(blh<400), 1.45, np.where((ws<8)&(blh<600), 1.20, 1.0))
        nov = np.where((dt.dt.year==2024)&(dt.dt.month==11)&(dt.dt.day<=10), 1.6, 1.0)
        
        aqi_base = sea * diu * wk * trp * nov
        aqi = np.clip(aqi_base + np.random.normal(0, aqi_base*0.08), 20, 500)
        pm25 = np.clip(aqi*0.55 + np.random.normal(0,8,n), 10, 350)
        pm10 = np.clip(pm25*np.random.uniform(1.5, 2.5, n), 15, 600)
        no2 = np.clip(aqi*0.12*np.where((hour>=8)&(hour<=10), 1.4, np.where((hour>=18)&(hour<=21), 1.3, 0.85)) + np.random.normal(0,5,n), 5, 200)
        so2 = np.clip(aqi*0.04*np.where((hour>=9)&(hour<=18)&(dow<5),1.5,0.7) + np.random.normal(0,3,n), 1, 80)
        co  = np.clip(aqi*0.008*np.where((month>=11)|(month<=2),1.6,1.0)*np.where(hour<=6,1.4,1.0) + np.random.normal(0,0.2,n), 0.3, 10)
        o3  = np.clip(60*np.where((hour>=12)&(hour<=17),1.8,0.4)*np.where((month>=3)&(month<=9),1.4,0.6) + np.random.normal(0,8,n), 5, 150)
        
        df['aqi'], df['pm25'], df['pm10'], df['no2'], df['so2'], df['co'], df['o3'] = aqi, pm25, pm10, no2, so2, co, o3
        df['hour'], df['dow'], df['month'] = hour, dow, month
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def clean_data(df):
    print("Cleaning & Engineering Features...")
    df = df.drop_duplicates(subset=['datetime', 'station']).sort_values(['station', 'datetime'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    # FIXED: Replaced method='time' with method='linear' since we don't have datetime as index.
    df[num_cols] = df.groupby('station')[num_cols].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    for col in ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    df['pm_ratio'] = df['pm10'] / (df['pm25'] + 1e-6)
    df['wind_sin'] = np.sin(np.radians(df['wind_direction']))
    df['wind_cos'] = np.cos(np.radians(df['wind_direction']))
    df['hour_sin'], df['hour_cos'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'], df['dow_cos'] = np.sin(2*np.pi*df['dow']/7), np.cos(2*np.pi*df['dow']/7)
    df['month_sin'], df['month_cos'] = np.sin(2*np.pi*df['month']/12), np.cos(2*np.pi*df['month']/12)
    return df

# ══════════════════════════════════════════════════════════════
# 2. XGBOOST 72-HOUR FORECASTER
# ══════════════════════════════════════════════════════════════
def train_forecaster(df):
    print("\nTraining XGBoost Forecaster (Multi-Step)...")
    st_df = df[df['station'] == 'Anand_Vihar'].sort_values('datetime').reset_index(drop=True)
    
    features = ['aqi', 'pm25', 'pm10', 'wind_speed', 'humidity', 'temperature', 'boundary_layer_height',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    
    st_df = st_df[features].dropna()
    scaler = MinMaxScaler()
    st_scaled = scaler.fit_transform(st_df)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "aqi_scaler.pkl"))
    
    LOOKBACK = 72
    HORIZON = 72
    target_idx = features.index('aqi')
    
    X, y = [], []
    total = len(st_scaled) - LOOKBACK - HORIZON + 1
    for i in range(total):
        X.append(st_scaled[i : i+LOOKBACK].flatten())
        y.append(st_scaled[i+LOOKBACK : i+LOOKBACK+HORIZON, target_idx])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    tr_size = len(X) - 1000
    X_tr, y_tr = X[:tr_size], y[:tr_size]
    X_te, y_te = X[tr_size:], y[tr_size:]
    
    base_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, n_jobs=-1, tree_method='hist')
    model = MultiOutputRegressor(base_xgb, n_jobs=1)
    
    model.fit(X_tr, y_tr)
    
    y_pred_scaled = model.predict(X_te)
    
    dummy_true = np.zeros((y_te.shape[0] * HORIZON, len(features)))
    dummy_pred = np.zeros((y_pred_scaled.shape[0] * HORIZON, len(features)))
    dummy_true[:, target_idx] = y_te.flatten()
    dummy_pred[:, target_idx] = y_pred_scaled.flatten()
    
    true_aqi = scaler.inverse_transform(dummy_true)[:, target_idx]
    pred_aqi = scaler.inverse_transform(dummy_pred)[:, target_idx]
    
    mae = mean_absolute_error(true_aqi, pred_aqi)
    print(f"XGBoost Forecaster trained! OVERALL MAE for 72h: {mae:.2f}")
    
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_forecaster.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "forecaster_features.pkl"))
    print("Saved forecaster models to /models/")

# ══════════════════════════════════════════════════════════════
# 3. XGBOOST SOURCE CLASSIFIER
# ══════════════════════════════════════════════════════════════
def apply_source_labels(df):
    labels = np.full(len(df), 'unclassified', dtype=object)
    
    trapped = (df['wind_speed']<4) & (df['boundary_layer_height']<400) & (df['aqi']>200)
    labels[trapped] = 'weather_trapped'
    
    burning = ~trapped & (df['co']>2.5) & (df['pm25']>100) & ((df['hour']>=21)|(df['hour']<=7)|((df['month']>=10)&(df['month']<=11)))
    labels[burning] = 'biomass_burning'
    
    industrial = ~trapped & ~burning & (df['so2']>20) & (df['hour']>=9) & (df['hour']<=18) & (df['dow']<5)
    labels[industrial] = 'industrial'
    
    dust = ~trapped & ~burning & ~industrial & (df['pm_ratio']>2.5) & (df['humidity']<50) & (df['hour']>=8) & (df['hour']<=20)
    labels[dust] = 'dust_construction'
    
    traffic = ~trapped & ~burning & ~industrial & ~dust & (df['no2']>60) & (((df['hour']>=8)&(df['hour']<=11))|((df['hour']>=17)&(df['hour']<=21)))
    labels[traffic] = 'traffic'
    
    rem = df[labels == 'unclassified'].copy()
    fb = pd.Series('traffic', index=rem.index)
    fb[rem['pm_ratio']>2.0] = 'dust_construction'
    fb[rem['so2']>15] = 'industrial'
    fb[rem['co']>1.5] = 'biomass_burning'
    labels[labels == 'unclassified'] = fb.values
    return labels

def train_classifier(df):
    print("\nTraining XGBoost Source Classifier...")
    df['source_label'] = apply_source_labels(df)
    
    features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'wind_speed', 'wind_sin', 'wind_cos',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'humidity', 'temperature',
                'pm_ratio', 'boundary_layer_height', 'surface_pressure']
    
    clf_data = df[features + ['source_label']].dropna()
    le = LabelEncoder()
    y = le.fit_transform(clf_data['source_label'])
    X = clf_data[features].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    weights = compute_sample_weight('balanced', y_tr)
    
    xgb_clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method='hist')
    xgb_clf.fit(X_tr, y_tr, sample_weight=weights)
    
    acc = accuracy_score(y_te, xgb_clf.predict(X_te))
    print(f"XGBoost Classifier trained! ACCURACY: {acc*100:.1f}%")
    
    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "source_classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "source_encoder.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "classifier_features.pkl"))
    print("Saved classification models to /models/")

# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    w_df = fetch_weather()
    a_df = generate_aqi(w_df)
    df = clean_data(a_df)
    
    train_forecaster(df)
    train_classifier(df)
    print("\nML Pipeline completed successfully! Models ready for FastAPI.")
