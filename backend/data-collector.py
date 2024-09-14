import sys
sys.path.append(r'C:\Users\vishn\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages')
import requests
import pymongo

def get_match_ids(season: str):
    try:
        s1 = season
        s2 = str(int(season) + 1)
        url = f"https://www.fotmob.com/api/leagues?id=87&ccode3=IND&season={s1}%2F{s2}"
        response = requests.get(url).json()
        data = response["matches"]["allMatches"]
        ids = [match["id"] for match in data]
        return ids
    except Exception as e:
        print(f"Error in get_data: {e}")
        return None


def insert_data(season: str):
    ids = get_match_ids(season)

    data = []
    
    for id in ids:
        try:
            url = f"https://www.fotmob.com/api/matchDetails?matchId={id}"
            response = requests.get(url).json()
            
        except Exception as e:
            print(f"Error in insert_data: {e}")

    if data:
        collection = db[f"season_{season}"]
        try:
            if isinstance(data, list):
                result = collection.insert_many(data)
                print(f"Inserted {len(result.inserted_ids)} records for season {season}")
            else:
                result = collection.insert_one(data)
                print(f"Inserted 1 record for season {season}")
        except Exception as e:
            print(f"Error inserting data: {e}")

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://vishnuluxx:OHqGeEuCn842o31i@cluster0.cqtwr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                             tls=True,
                            tlsAllowInvalidCertificates=True)
db = client["football_data"]    

# Insert season data
season = "2023"
insert_data(season)
