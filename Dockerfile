FROM apache/airflow:2.9.0
USER 50000:0
RUN pip install --upgrade pip
RUN pip install --no-cache-dir uv
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt
