# fake_iot_sensor.py
from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/iot/latest')
def latest():
    # Simulate sensor readings for proof-of-concept
    return jsonify({
        "temp_C": round(random.uniform(22, 32), 2),
        "humidity_pct": round(random.uniform(40, 90), 2),
        "rainfall_mm": round(random.uniform(0, 300), 1),
        "solar_MJ": round(random.uniform(12, 22), 2)
    })

if __name__ == '__main__':
    app.run(port=5001)
