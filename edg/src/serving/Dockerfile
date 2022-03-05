FROM python:3.7-slim
WORKDIR /app

COPY serving/main.py \
    requirements.txt \
    src/training/simple_linear_regr_utils.py \
    src/training/simple_linear_regr.py \
    model/model.npy \
    ./

RUN pip install --upgrade -r requirements.txt

ENTRYPOINT ["python"]
CMD ["/app/main.py"]
