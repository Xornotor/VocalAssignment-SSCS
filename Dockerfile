FROM tensorflow/tensorflow:2.14.0-gpu-jupyter
WORKDIR /app
COPY requirements.txt .
RUN apt -y update && apt -y install libsndfile1 build-essential cmake && apt -y clean
RUN pip install -r requirements.txt && pip cache purge
RUN rm requirements.txt
EXPOSE 8888 
EXPOSE 6006

#ARG UID=10001
#RUN adduser \
    #--disabled-password \
    #--gecos "" \
    #--home "/appuser" \
    #--shell "/sbin/nologin" \
    #--uid "${UID}" \
    #appuser

#RUN chmod -R 777 /app

#USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
