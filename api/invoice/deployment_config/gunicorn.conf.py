# This python script gets all env variables starting with 'GUNICORN_' and set the corresponding parameters for execution

import os

for k, v in os.environ.items():
    if k.startswith("GUNICORN_"):
        key = k.split('_', 1)[1].lower()
        locals()[key] = v