FROM python:3.7-slim

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

ENV GIT_PYTHON_REFRESH=quiet

RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libgl1 libglib2.0-0 libxrender-dev
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt

COPY . /fastapi

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]