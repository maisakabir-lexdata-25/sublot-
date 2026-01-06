from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Port 8501 is accessible!"

if __name__ == '__main__':
    print("Starting diagnostic server on port 8501...")
    app.run(port=8501, host='127.0.0.1')
