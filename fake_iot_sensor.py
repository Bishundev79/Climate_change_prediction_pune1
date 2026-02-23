"""
Fake IoT Sensor — REST API demo.

Simulates a weather sensor station for Pune, returning random
but realistic climate readings.  Used by the Forecast page's
"IoT Sensor (REST)" data source.

Usage:
    python fake_iot_sensor.py        # starts on http://localhost:5001
    curl localhost:5001/iot/latest    # JSON sensor reading
"""

try:
    from flask import Flask, jsonify
except ImportError:
    raise SystemExit(
        "Flask is required: pip install flask\n"
        "Or install all deps: pip install -r requirements.txt"
    )
import random

app = Flask(__name__)

def get_sensor_data():
    """Generate random sensor readings"""
    return {
        "temp_C": round(random.uniform(22, 32), 2),
        "humidity_pct": round(random.uniform(40, 90), 2),
        "rainfall_mm": round(random.uniform(0, 300), 1),
        "solar_MJ": round(random.uniform(12, 22), 2)
    }

@app.route('/')
def home():
    """Display sensor data on a simple HTML page"""
    data = get_sensor_data()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IoT Sensor Data</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .sensor-data {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .data-item {{
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid #4CAF50;
                background-color: #f9f9f9;
            }}
            .label {{
                font-weight: bold;
                color: #555;
            }}
            .value {{
                float: right;
                color: #2196F3;
                font-size: 18px;
            }}
            .note {{
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>🌡️ IoT Sensor Dashboard</h1>
        <div class="sensor-data">
            <div class="data-item">
                <span class="label">Temperature:</span>
                <span class="value">{data['temp_C']}°C</span>
            </div>
            <div class="data-item">
                <span class="label">Humidity:</span>
                <span class="value">{data['humidity_pct']}%</span>
            </div>
            <div class="data-item">
                <span class="label">Rainfall:</span>
                <span class="value">{data['rainfall_mm']} mm</span>
            </div>
            <div class="data-item">
                <span class="label">Solar Radiation:</span>
                <span class="value">{data['solar_MJ']} MJ/m²</span>
            </div>
        </div>
        <p class="note">Page auto-refreshes every 30 seconds | API endpoint: <a href="/iot/latest">/iot/latest</a></p>
    </body>
    </html>
    """
    return html

@app.route('/iot/latest')
def latest():
    # Simulate sensor readings for proof-of-concept
    return jsonify(get_sensor_data())

if __name__ == '__main__':
    app.run(port=5001)
