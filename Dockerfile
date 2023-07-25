FROM pytorch/pytorch
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY . /app/
EXPOSE 5000
CMD python ./app.py