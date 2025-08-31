#!/bin/bash

echo "🚀 Deploying Crypto Mining Platform API..."

# Check Python version
python3 --version

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Run database migrations (if needed)
echo "🗄️ Setting up database..."
python3 -c "
from server import create_tables, get_db, create_default_data
create_tables()
db = next(get_db())
create_default_data(db)
print('Database setup complete')
"

# Start the server
echo "🌐 Starting server..."
if [ "$1" = "dev" ]; then
    python3 server.py
else
    gunicorn server:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120
fi
