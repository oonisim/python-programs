#!/bin/bash

if [ -z "${WSGI_MODULE}" ] || [ -z "${WSGI_APP}" ]
then
      echo "WSGI env variable missing, Gunicorn wont start."
      exit 0
fi

gunicorn --config ./deployment_config/gunicorn.conf.py  --chdir=/project ${WSGI_MODULE}:${WSGI_APP}
