from flask import Flask
import sys
sys.path.append("TSP_project")
from gui import create_dash_app

app = Flask(__name__)
server = app.server

create_dash_app(app)

if __name__ == "__main__":
    app.run(debug=True)