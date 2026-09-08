# Imagem base leve com Python 3.12
FROM python:3.12-slim

# Evita arquivos .pyc e mantém logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define a raiz do projeto para imports Python
ENV PYTHONPATH=/app

# Diretório de trabalho do container
WORKDIR /app

# Copia primeiro as dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Porta utilizada pelo Streamlit
EXPOSE 8501

# Inicializa a aplicação
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# Build:
# docker build -t tech-challenge-fase3-literacy-ml .

# Execução:
# docker run -p 8501:8501 tech-challenge-fase3-literacy-ml