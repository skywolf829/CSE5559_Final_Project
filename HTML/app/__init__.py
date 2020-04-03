from flask import Flask
app = Flask(__name__)

from app import routes

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)