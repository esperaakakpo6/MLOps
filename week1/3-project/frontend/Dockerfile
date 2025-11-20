# ----- Image de base Python adaptée aux gros requirements -----
FROM python:3.10-slim

# ----- Installer les dépendances système critiques -----
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libhdf5-dev \
    libxml2 \
    libxml2-dev \
    libxslt-dev \
    libsqlite3-dev \
    gdal-bin \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----- Définir le dossier de travail -----
WORKDIR /app

# ----- Copier le requirements.txt -----
COPY requirements.txt .

# ----- Mettre à jour pip -----
RUN pip install --upgrade pip

# ----- Installer toutes les dépendances (très long !) -----
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copier tout ton code frontend -----
COPY . .

# ----- Exposer le port utilisé dans Docker -----
EXPOSE 8080

# ----- Commande pour lancer Streamlit -----
CMD ["streamlit", "run", "frontend.py", "--server.port=8080", "--server.address=0.0.0.0"]
