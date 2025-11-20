# ----- Base Python très stable pour gros requirements -----
FROM python:3.10-slim

# ----- Installer les dépendances système nécessaires -----
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----- Dossier d'app -----
WORKDIR /app

# ----- Copier le requirements.txt -----
COPY requirements.txt .

# ----- Mise à jour pip -----
RUN pip install --upgrade pip

# ----- Installer toutes les dépendances (très long !) -----
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copier tout ton backend (y compris modele/) -----
COPY . .

# ----- Exposer le port (défini dans backend.py) -----
EXPOSE 8000

# ----- Lancer l’API FastAPI -----
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]