FROM python:3.9

ARG inputParam
ENV inputCommand=$inputParam
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
# RUN python -m pip install --upgrade 'tensorflow_data_validation[visualization]<2'

RUN chmod -R 755 /app 

CMD echo "alias ht='/app/launcher.sh ht'" >> ~/.bashrc