FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8050
CMD ["streamlit", "run", "app.py", "--server.port=8050", "--server.address=0.0.0.0"]


