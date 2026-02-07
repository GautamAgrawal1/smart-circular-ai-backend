from geo_utils import haversine
from places import places

def recommend_places(action, user_lat, user_lon):

    if action == "Repair":
        places = [
            {
                "name": "Green Repair Hub",
                "lat": 25.3189,
                "lon": 82.9751
            }
        ]
    elif action == "Recycle":
        places = [
            {
                "name": "EcoRecycle Plant",
                "lat": 25.3292,
                "lon": 82.9613
            }
        ]
    else:
        places = [
            {
                "name": "Reuse Market",
                "lat": 25.3144,
                "lon": 82.9821
            }
        ]

    # compute distance
    out = []
    for p in places:
        d = haversine(user_lat, user_lon, p["lat"], p["lon"])
        out.append({
            "name": p["name"],
            "lat": p["lat"],
            "lon": p["lon"],
            "distance_km": round(d,2)
        })

    return out

