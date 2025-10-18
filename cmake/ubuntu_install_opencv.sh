#!/bin/bash

set -e  # Exit on error

if [ "$EUID" -ne 0 ]; then
  echo "[!] Please run as root (use sudo)"
  exit 1
fi

echo "[+] Updating system..."
apt update && apt upgrade -y

echo "[+] Installing OpenCV dependencies..."
apt install -y build-essential cmake git pkg-config \
libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev \
libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
python3-dev python3-pip

echo "[+] Installing OpenCV for Python via pip..."
pip3 install opencv-python opencv-python-headless

cd /opt
[ -d "opencv" ] || git clone https://github.com/opencv/opencv.git
[ -d "opencv_contrib" ] || git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir -p build && cd build

echo "[+] Configuring build with CMake..."
cmake -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      -DWITH_TBB=ON -DWITH_V4L=ON -DWITH_OPENGL=ON -DWITH_GTK=ON \
      -DBUILD_EXAMPLES=ON -DBUILD_opencv_python3=ON ..

echo "[+] Compiling OpenCV... (grab a coffee)"
make -j$(nproc)

echo "[+] Installing OpenCV..."
make install
ldconfig

echo "[✓] OpenCV successfully installed for C++ and Python."
