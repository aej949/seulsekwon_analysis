import requests
import json

KEY = "4e7a4a4d70646b73343261564e4c67"
SERVICES = [
    "SeoulPoliceStationWGS",
    "SeoulPharmacyStatusInfo",
    "SeoulWomensSafeDelivery"
]

def check_service(service):
    url = f"http://openapi.seoul.go.kr:8088/{KEY}/json/{service}/1/5/"
    try:
        resp = requests.get(url)
        data = resp.json()
        print(f"--- {service} ---")
        if service in data:
            rows = data[service]['row']
            if rows:
                print(f"Sample keys: {rows[0].keys()}")
                print(f"Sample row: {rows[0]}")
            else:
                print("No rows.")
        else:
            print(f"Error: {data}")
    except Exception as e:
        print(f"Failed: {e}")

for s in SERVICES:
    check_service(s)
