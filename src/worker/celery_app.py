import os
from celery import Celery

# Read the URLs from environment variables (used in Docker), fallback to localhost for local testing
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend_url = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'credit_tasks',
    broker=broker_url,
    backend=backend_url,
    include=['src.worker.tasks']  # Explicitly point to where your tasks live
)

# Optional: Basic configuration for enterprise logging and timezone
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)