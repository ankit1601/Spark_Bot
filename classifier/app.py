__author__ = 'TI307867'

from flask import Flask
from flask import request
app = Flask(__name__)
from flask import jsonify
from flask_cors import CORS, cross_origin
cors = CORS(app)

import submission_classification_Toxic as cls

@app.route('/train')
def train():
    cls.train_model()
    return "Success!", 200

@app.route('/get/sent')
@cross_origin()
def get_sent():
    text = request.args.get('text')
    model, prob = cls.predict(text)
    return jsonify({"res": model[prob.index(max(prob))] }), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=True)