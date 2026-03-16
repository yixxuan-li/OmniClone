#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo " Step 1/5: Packaging teleop environment and ROS 2"
echo "=========================================="
if [ ! -f teleop_env.tar.gz ]; then
    tar -czf teleop_env.tar.gz -C /home/unitree/.local/share/mamba/envs/teleop .
else
    echo "  teleop_env.tar.gz (already exists, skipping)"
fi

if [ ! -f ros_foxy.tar.gz ]; then
    echo "  Packaging /opt/ros/foxy ..."
    tar -czf ros_foxy.tar.gz -C /opt/ros foxy
else
    echo "  ros_foxy.tar.gz (already exists, skipping)"
fi

if [ ! -f unitree_ros2_ws.tar.gz ]; then
    echo "  Packaging /home/unitree/project/public/unitree_ros2 ..."
    tar -czf unitree_ros2_ws.tar.gz -C /home/unitree/project/public unitree_ros2
else
    echo "  unitree_ros2_ws.tar.gz (already exists, skipping)"
fi

echo "=========================================="
echo " Step 2/5: Packaging unitree_sdk2_python and numpy"
echo "=========================================="
if [ ! -f unitree_sdk2_python.tar.gz ]; then
    tar -czf unitree_sdk2_python.tar.gz -C /home/unitree/project unitree_sdk2_python
else
    echo "  unitree_sdk2_python.tar.gz (already exists, skipping)"
fi

if [ ! -f local_site_packages.tar.gz ]; then
    tar -czf local_site_packages.tar.gz -C /home/unitree/.local/lib/python3.8 site-packages
else
    echo "  local_site_packages.tar.gz (already exists, skipping)"
fi

if [ ! -f PyQt5.tar.gz ]; then
    tar -czf PyQt5.tar.gz -C /usr/lib/python3/dist-packages PyQt5 sip.cpython-38-aarch64-linux-gnu.so sipconfig.py sipdistutils.py
else
    echo "  PyQt5.tar.gz (already exists, skipping)"
fi

echo "=========================================="
echo " Step 3/5: Copying system shared libraries"
echo "=========================================="
if [ ! -d system_libs ] || [ -z "$(ls -A system_libs)" ]; then
    mkdir -p system_libs
    ldd /home/unitree/tools/XRoboToolkit-Orin-Video-Sender/OrinVideoSender \
        | awk '/=>/ {print $3}' | grep -v '^$' \
        | while read lib; do cp -L "$lib" system_libs/ 2>/dev/null; done
    
    # ─── Copy CycloneDDS libraries required by inspire_hand_sdk ───
    if [ -d "/home/unitree/unitree_ros2/cyclonedds_ws/install/cyclonedds/lib" ]; then
        cp -L /home/unitree/unitree_ros2/cyclonedds_ws/install/cyclonedds/lib/libddsc.so* system_libs/ 2>/dev/null || true
    elif [ -d "/opt/ros/foxy/lib" ]; then
        cp -L /opt/ros/foxy/lib/aarch64-linux-gnu/libddsc.so* system_libs/ 2>/dev/null || true
    fi
    cp -L /usr/lib/aarch64-linux-gnu/libssl.so.1.1 system_libs/ 2>/dev/null || true
    cp -L /usr/lib/aarch64-linux-gnu/libcrypto.so.1.1 system_libs/ 2>/dev/null || true
    
    # ─── Copy ROS 2 system shared libraries ───
    ldd /opt/ros/foxy/lib/*.so /opt/ros/foxy/lib/python3.8/site-packages/rclpy/*.so 2>/dev/null \
        | awk '/=>/ {print $3}' | grep -v '^$' | grep -E "^/usr/lib|^/lib" | grep -v "tegra/" | sort -u \
        | while read lib; do cp -n -L "$lib" system_libs/ 2>/dev/null; done

    # ─── Copy ALL Python site-packages system libraries (including Numpy, PyTorch, PyTorch_Kinematics, etc) ───
    find /home/unitree/.local/share/mamba/envs/teleop/lib/python3.8/site-packages /home/unitree/.local/lib/python3.8/site-packages -name "*.so" -type f | xargs ldd 2>/dev/null \
        | awk '/=>/ {print $3}' | grep -v '^$' | grep -E "^/usr/local/cuda|^/usr/lib/aarch64-linux-gnu|^/lib/aarch64-linux-gnu" | grep -v "tegra/" | sort -u \
        | while read lib; do cp -n -L "$lib" system_libs/ 2>/dev/null || true; done

    # ─── Copy PyQt5 shared libraries ───
    ldd /usr/lib/python3/dist-packages/PyQt5/*.so 2>/dev/null \
        | awk '/=>/ {print $3}' | grep -v '^$' | sort -u \
        | while read lib; do cp -L "$lib" system_libs/ 2>/dev/null; done

    # ─── Copy GStreamer plugins and their dependencies ───
    if [ -d /usr/lib/aarch64-linux-gnu/gstreamer-1.0 ]; then
        mkdir -p system_libs/gstreamer-1.0
        cp -L /usr/lib/aarch64-linux-gnu/gstreamer-1.0/*.so system_libs/gstreamer-1.0/ 2>/dev/null || true
        find /usr/lib/aarch64-linux-gnu/gstreamer-1.0 -name "*.so" -type f | xargs ldd 2>/dev/null \
            | awk '/=>/ {print $3}' | grep -v '^$' | sort -u \
            | while read lib; do cp -n -L "$lib" system_libs/ 2>/dev/null || true; done
    fi

    # ─── Copy ffmpeg and dependencies ───
    if command -v ffmpeg > /dev/null; then
        cp -L $(which ffmpeg) system_libs/ 2>/dev/null || true
        ldd $(which ffmpeg) 2>/dev/null | awk '/=>/ {print $3}' | grep -v '^$' | sort -u \
            | while read lib; do cp -n -L "$lib" system_libs/ 2>/dev/null || true; done
    fi

else
    echo "  (already exists, skipping)"
fi

echo "=========================================="
echo " Step 4/5: Staging external tools"
echo "=========================================="
if [ ! -d "XRoboToolkit-Orin-Video-Sender" ]; then
    cp -r /home/unitree/tools/XRoboToolkit-Orin-Video-Sender .
else
    echo "  XRoboToolkit-Orin-Video-Sender (already staged)"
fi
if [ ! -d "inspire_hand_sdk" ]; then
    cp -r /home/unitree/tools/inspire_hand_ws/inspire_hand_sdk .
else
    echo "  inspire_hand_sdk (already staged)"
fi

echo "=========================================="
echo " Step 5/5: Building Docker image"
echo "=========================================="
sg docker -c "docker build -t omniclone-release:latest -f docker/Dockerfile_omniclone ."

echo ""
echo "=========================================="
echo " SUCCESS: omniclone-release:latest created!"
echo "=========================================="
echo ""
echo "Run the container with:"
echo "  docker run -it --net=host --ipc=host omniclone-release:latest"
echo ""
echo "Test OrinVideoSender:"
echo "  docker run -it --net=host --ipc=host --privileged -v /dev:/dev --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all omniclone-release:latest bash -c 'cd /home/unitree/tools/XRoboToolkit-Orin-Video-Sender && ./open_image_sender.sh'"
echo ""
echo "Test inspire_hand driver:"
echo "  docker run -it --net=host omniclone-release:latest python /home/unitree/tools/inspire_hand_ws/inspire_hand_sdk/example/Headless_driver_double.py"
echo ""
echo "(Optional) Clean up staging files:"
echo "  rm -rf teleop_env.tar.gz unitree_sdk2_python.tar.gz system_libs/ XRoboToolkit-Orin-Video-Sender/ inspire_hand_sdk/"
