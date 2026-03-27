from fastapi.testclient import TestClient
from app.main import app
import pprint

client = TestClient(app)

def run_tests():
    print("\n--- Testing Phase 4 Admin Dashboard Endpoints ---")
    
    # 1. Test Single Source Policy lookup
    print("\n[1] Testing Standalone Policy Lookup (Traffic):")
    res = client.get("/api/v1/admin/policies/traffic")
    if res.status_code == 200:
        print("✅ SUCCESS")
        pprint.pprint(res.json())
    else:
        print(f"❌ FAILED: {res.text}")

    # 2. Test Live What-If Simulator
    print("\n[2] Testing Live 'What-If' Edge Simulator (Ban Heavy Vehicles in Anand Vihar @ 80% Enforcement):")
    sim_payload = {
        "ward_name": "Anand Vihar",
        "policy_action": "ban_heavy_vehicles",
        "severity": 80.0
    }
    res = client.post("/api/v1/admin/simulate", json=sim_payload)
    if res.status_code == 200:
        print("✅ SUCCESS")
        pprint.pprint(res.json())
    else:
        print(f"❌ FAILED: {res.text}")

    # 3. Test Updated Public Ward Endpoint (Does it include actionable_policies?)
    print("\n[3] Testing Public Ward Endpoint (Checking for injected Phase 4 policies):")
    res = client.get("/api/v1/aqi/ward/ITO")
    if res.status_code == 200:
        print("✅ SUCCESS")
        data = res.json()
        print(f"Actionable Policies Found: {len(data.get('actionable_policies', []))}")
        print("Policies:", data.get('actionable_policies'))
    else:
        print(f"❌ FAILED: {res.text}")

if __name__ == "__main__":
    run_tests()
