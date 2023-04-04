import datetime
from flask import Flask, render_template, url_for, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import requests
import json
from urllib.request import urlopen, Request
from firebase_admin import db, credentials, firestore, initialize_app
# import urllib3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from classify import *

app = Flask(__name__)
CORS(app)

gsr_model = pickle.load(open('gsr.pickle', 'rb'))
# test_data = np.array([[93, 41, 40, 20.87, 82.032, 6.5, 205.9]])
# prediction = crop_model.predict(test_data)
# print(prediction)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'databaseURL': 'https://gsrwithuv-default-rtdb.firebaseio.com/'
})
# db = firestore.client()
# gsr_ref = db.reference("/gsr")
gsr_coll = firestore.client().collection('gsr_data')

datasetRef = db.reference('/dataset')
# print("Get Data", datasetRef.get())

dataDict = dict()
currDataLen = 0

def checkUpdate():
    global dataDict
    global currDataLen
    newLen = len(dataDict)
    if currDataLen != newLen:
        currDataLen = newLen
        # TODO: call classification
        print("newDict:", dataDict)

def listener(event):
    global dataDict
    print(event.event_type)  # can be 'put' or 'patch'
    # print(event.path)  # relative to the reference, it seems
    if event.data == None:
        print("Data is empty")
        return None
    # print(event.data)  # new data at /reference/event.path. None if deleted
    # print(type(event.data))
    # print(event.data.keys())
    newData = event.data
    print(newData)
    for key in newData.keys():
        dataDict[key] = newData[key]
    checkUpdate()
    # print(datasetRef.order_by_key().limit_to_last(10).get())
    # print(db.Query(limit_to_last=10).reference('/dataset'))

datasetRef.listen(listener)

# print(datasetRef.get_if_changed("/dataset"))


def getCityWeather(name):
    baseURL = "http://api.openweathermap.org/data/2.5/weather?"
    apiKey = "25ab0f2df8aee2f1e4def94d33a8900b"
    cityWeatherURL = baseURL + "appid=" + apiKey + "&q=" + name
    jsonResponse = requests.get(cityWeatherURL).json()
    if jsonResponse["cod"] != 200:
        return None
    else:
        cityData = jsonResponse["main"]
        temperature = round((cityData["temp"] - 273.15), 2)
        humidity = cityData["humidity"]
        return temperature, humidity


@app.route('/')
def index():
    # print(getCityWeather("Tiruppur"))
    return render_template('index.html')
    # return render_template('prediction_result.html', predicted_crop='orange')

@app.route('/getGsrStatus')
def getGsrStatus():
    global dataDict
    gsr_value_without_connection = request.args.get("initialValue") or 252
    gsr_value_without_connection = float(gsr_value_without_connection)
    gsr_value_without_connection = max(33, gsr_value_without_connection)
    print("In Route", dataDict)
    if dataDict == None or len(dataDict) == 0:
        return "<h1>No Data</h1>"
    dataDict = datasetRef.get()
    # print(dataDict.values())
    sample_array = np.fromiter(map(lambda d: float(d["gsrValue"]), dataDict.values()), dtype=float)
    ans = dtw_algorithm(gsr_value_without_connection, sample_array)
    print(ans)
    print("---------------------------------------------")
    print(ans[0][1])
    print("---------------------------------------------")
    return str(dtw_algorithm(gsr_value_without_connection, sample_array))


@app.route('/getGsrResult')
def getGsrResult():
    print("working")
    global dataDict
    gsr_value_without_connection = request.args.get("initialValue") or 252
    gsr_value_without_connection = float(gsr_value_without_connection)
    gsr_value_without_connection = max(33, gsr_value_without_connection)
    print("In Route", dataDict)
    # if dataDict == None or len(dataDict) == 0:
    #     return jsonify({"status":"no input"})
    dataDict = datasetRef.get()
    # print(dataDict.values())
    sample_array = np.fromiter(map(lambda d: float(d["gsrValue"]), dataDict.values()), dtype=float)
    ans = dtw_algorithm(gsr_value_without_connection, sample_array)
    # print(ans)
    # print("---------------------------------------------")
    # print(ans[0][1])
    print("result")
    return jsonify({"status":ans[0][1]})
    # print("---------------------------------------------")
    # return str(dtw_algorithm(gsr_value_without_connection, sample_array))



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    age = request.args.get('age') or 21
    name = request.args.get('name') or 'John Doe'
    gsrValue = request.args.get('gsrValue') or 270
    lat = request.args.get('lat') or "10.900"
    lng = request.args.get('lng') or "76.904"
    print(lat, lng)
    # Get gsr from arduino
    gsrValue, uv_max, uv = 0, 0, 0
    try:
        url = "http://192.168.77.201/getgsr"
        response = urlopen(url)
        data = response.read()
        dt = json.loads(data)
        print(dt)
        gsrValue = dt["gsrAvg"]
    except:
        gsrValue = 270

    # Get uv value
    try:
        req = Request(f"https://api.openuv.io/api/v1/uv?lat={lat}&lng={lng}", None, {'x-access-token' : '7044ee19687d7cffd25d10a64d540b50'})
        response = urlopen(req)
        dt = json.loads(response.read())
        # print(dt)
        uv_max = dt["result"]["uv_max"]
        uv = dt["result"]["uv"]
    except:
        uv_max = 8
        uv = 8
    print(uv, uv_max)


    data = np.array([[age, gsrValue]])
    predicted_status = gsr_model.predict(data)
    print(predicted_status)
    res = {
        "name": name,
        "age": age,
        "gsrValue": gsrValue,
        "status": predicted_status[0],
        "uv": uv,
        "uv_max": uv_max,
        "lat": lat,
        "lng": lng,
        "timestamp": datetime.datetime.now(),
    }
    print(res)
    # gsr_ref.push().set(res)
    gsr_coll.add(res)
    # return predicted_status[0]
    return jsonify(res)



# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         N = int(request.form['nitrogen'])
#         P = int(request.form['phosphorous'])
#         K = int(request.form['pottasium'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
#         city = request.form.get('city')
#         print(N, P, K, ph, rainfall, city)
#         try:
#             tempWeather = getCityWeather(city)
#             temperature = 28
#             humidity = 83
#             if tempWeather != None:
#                 temperature, humidity = tempWeather
#             data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
#             predicted_crop = crop_model.predict(data)
#             return render_template('prediction_result.html', predicted_crop=predicted_crop[0])
#         except:
#             print("Error")
#     return render_template('prediction_form.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
