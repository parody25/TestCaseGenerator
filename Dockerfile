FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8504
CMD ["streamlit", "run", "app.py", "--server.port=8504", "--server.address=0.0.0.0"]
