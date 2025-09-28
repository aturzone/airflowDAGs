#!/bin/bash
echo "🧹 شروع پاک‌سازی کامل Docker..."

# پاک‌سازی کامل
sudo systemctl stop docker
sudo pkill -f docker
sudo rm -rf /var/lib/docker/containers/* 2>/dev/null || true
sudo systemctl start docker
sudo docker system prune -af --volumes

# تست Docker
sudo docker run hello-world

# تنظیمات سیستم
sudo sysctl vm.overcommit_memory=1
sudo usermod -aG docker $USER

# به‌روزرسانی Airflow version
sed -i 's/apache\/airflow:2.8.1-python3.11/apache\/airflow:2.9.0/g' docker-compose.yml

# پاک‌سازی محلی
sudo rm -rf logs/* || true
mkdir -p logs dags plugins config

# دانلود images جدید
sudo docker-compose pull

# راه‌اندازی مرحله‌ای
sudo docker-compose up -d postgres redis
sleep 30

# تست اتصالات
docker-compose exec postgres pg_isready -U airflow
docker-compose exec redis redis-cli ping

# Airflow init
sudo docker-compose run --rm airflow-init

# شروع همه services
sudo docker-compose up -d

echo "✅ راه‌اندازی کامل شد!"
echo "🌐 Airflow: http://localhost:8080"
