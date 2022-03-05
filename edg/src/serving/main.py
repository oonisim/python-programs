import json
import logging
import numpy as np
from flask import (
    Flask,
    request,
    jsonify
)
from simple_linear_regr import (
    TYPE_FLOAT,
    SimpleLinearRegression
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)

# --------------------------------------------------------------------------------
# Load web runtime
# --------------------------------------------------------------------------------
app = Flask('prediction')

# --------------------------------------------------------------------------------
# Load Model
# --------------------------------------------------------------------------------
model = SimpleLinearRegression()
model.load("model.npy")


@app.route('/', methods=['GET'])
def health():
    return "OK"


def predict(data):
    X = np.array(data).astype(TYPE_FLOAT).reshape(-1, 1)
    y_hat = np.squeeze(model.predict(X))
    return json.dumps(y_hat.tolist())


def handle_request():
    if request.is_json:
        if request.content_length > 0:
            data = request.get_json()
            logging.debug("predict(): Payload:[\n%s\n]" % data)

            try:
                result = predict(data)
                return result, 200
            except Exception as e:
                logging.error("handle_request(): exception %s" % e)
                return "Internal Error", 500

        else:
            return "Expected POST data payload, got none.", 400
    else:
        return "Expected content-type application/json, got {}".format(request.content_type), 400


@app.route('/stream', methods=['POST'])
def stream():
    """Single prediction"""
    logging.debug(
        "stream(): request.content_type: [%s] content_length [%s]." %
        (request.content_type, request.content_length)
    )
    return handle_request()


@app.route('/batch', methods=['POST'])
def batch():
    """Batch prediction"""
    logging.debug(
        "batch(): request.content_type: [%s] content_length [%s]." %
        (request.content_type, request.content_length)
    )
    return handle_request()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
