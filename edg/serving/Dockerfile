FROM python:3.7-slim
WORKDIR /app

COPY [ \
    "requirements.txt", \
    "serving/main.py", \
    "training/simple_linear_regr_utils.py",\
     "training/simple_linear_regr.py", \
     "./" \
]
RUN pip install --upgrade -r requirements.txt

ENTRYPOINT ["python"]
CMD ["/app/main.py"]
