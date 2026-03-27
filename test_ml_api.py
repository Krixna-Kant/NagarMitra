from fastapi.testclient import TestClient
from app.main import app
import pprint

client = TestClient(app)

def run_tests():
    print("\n--- Testing Phase 2 ML Endpoints ---")
    
    # 1. Test WHRI Calculator GET
    print("\n[1] Testing WHRI Calculator:")
    response = client.get("/api/v1/ml/whri/Anand_Vihar?aqi_score=350&dengue_risk=60&heatwave_risk=40")
    if response.status_code == 200:
        print("✅ SUCCESS")
        pprint.pprint(response.json())
    else:
        print(f"❌ FAILED: {response.text}")

    # 2. Test Source Classifier POST
    print("\n[2] Testing Pollution Source Classifier:")
    source_payload = {
        "pm25": 120.0, "pm10": 350.0, "no2": 40.0, "so2": 10.0,
        "co": 1.5, "o3": 45.0, "wind_speed": 15.0, "wind_sin": 0.5, "wind_cos": 0.8,
        "hour_sin": -0.7, "hour_cos": -0.7, "dow_sin": 0.0, "dow_cos": 1.0,
        "humidity": 30.0, "temperature": 32.0, "pm_ratio": 3.0, 
        "boundary_layer_height": 800.0, "surface_pressure": 1010.0
    }
    response = client.post("/api/v1/ml/source/classify", json=source_payload)
    if response.status_code == 200:
        print("✅ SUCCESS")
        pprint.pprint(response.json())
    else:
        print(f"❌ FAILED: {response.text}")

    # 3. Test 72h Forecaster POST
    print("\n[3] Testing 72h AQI Forecaster:")
    # Build dummy 72-hour historical data arrays
    forecast_payload = {
        "aqi": [250.0] * 72,
        "pm25": [120.0] * 72,
        "pm10": [200.0] * 72,
        "wind_speed": [10.0] * 72,
        "humidity": [50.0] * 72,
        "temperature": [25.0] * 72,
        "boundary_layer_height": [600.0] * 72,
        "hour_sin": [0.0] * 72,
        "hour_cos": [1.0] * 72,
        "dow_sin": [0.0] * 72,
        "dow_cos": [1.0] * 72,
        "month_sin": [0.0] * 72,
        "month_cos": [1.0] * 72
    }
    response = client.post("/api/v1/ml/forecast/72h", json=forecast_payload)
    if response.status_code == 200:
        print("✅ SUCCESS")
        data = response.json()
        print(f"Horizon: {data['horizon']}h")
        print(f"Average Predicted AQI: {data['average_predicted_aqi']}")
        print(f"Peak Predicted AQI: {data['peak_predicted_aqi']}")
        print(f"Length of forecast array: {len(data['forecast_aqi'])}")
    else:
        print(f"❌ FAILED: {response.text}")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"Error during tests: {e}")
