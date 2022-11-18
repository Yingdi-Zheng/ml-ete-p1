FROM python:3.9
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN python -m pip install --upgrade 'tensorflow_data_validation[visualization]<2'


ENTRYPOINT ["/app/launcher.sh"]