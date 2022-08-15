Run locally
-----------

The server is started by this command:

`$ gunicorn -b 0.0.0.0:5000 run:app`


Testing
-------

To run the tests:

`$ pytest -v -W ignore::DeprecationWarning -l --tb=line -x`

This command will run the tests while ignoring deprecation warnings (usually coming from dependencies) and printing errors on a single line.


Run with docker
---------------

You can execute the build command from the root of the repository:
`$ docker-compose build`

Before running the containers, it is recommended to double check the env variables contained in var.env file.

You can now run the container:
``$ docker-compose up``

You can now browse http://localhost/api/doc :)
