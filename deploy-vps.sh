#!/bin/bash
# ============================================================
# سكريبت النشر اليدوي على VPS (بدون Docker)
# ============================================================
#   يشغّل مباشرة على الخادم:
#       chmod +x deploy-vps.sh && sudo ./deploy-vps.sh
# ============================================================

set -e

APP_DIR="${APP_DIR:-/opt/face-recognition-api}"
SERVICE_NAME="${SERVICE_NAME:-face-recognition-api}"
VENV_DIR="$APP_DIR/venv"

echo "========================================================"
echo "  🚀 نشر Face Recognition API"
echo "========================================================"

# 1️⃣ إنشاء المجلد
echo ""
echo "📁 [1/5] إنشاء المجلد..."
mkdir -p "$APP_DIR"

# 2️⃣ نسخ الملفات
echo ""
echo "📂 [2/5] نسخ الملفات..."
rsync -av --delete \
    --exclude='venv/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='.env' --exclude='test_images/' \
    ./ "$APP_DIR/"

# 3️⃣ إعداد البيئة الافتراضية والحزم
echo ""
echo "🐍 [3/5] إعداد البيئة الافتراضية..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

# 4️⃣ تثبيت مكتبات النظام
echo ""
echo "🔧 [4/5] تثبيت مكتبات النظام..."
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# 5️⃣ إعداد systemd service
echo ""
echo "⚙️ [5/5] إعداد systemd service..."
cat > "/etc/systemd/system/$SERVICE_NAME.service" << SERVICE
[Unit]
Description=Face Recognition API (FastAPI)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
Environment="PORT=8001"
Environment="HOST=0.0.0.0"

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "========================================================"
echo "  ✅ تم النشر بنجاح!"
echo "  🌐 http://localhost:8001"
echo "========================================================"

# اختبار
sleep 4
echo ""
echo "🔍 حالة الخدمة:"
systemctl status "$SERVICE_NAME" --no-pager
echo ""
echo "🏥 اختبار سريع:"
curl -s http://localhost:8001/ | python3 -m json.tool 2>/dev/null || \
    curl -s http://localhost:8001/ || \
    echo "⚠️  النموذج لا يزال قيد التحميل"
