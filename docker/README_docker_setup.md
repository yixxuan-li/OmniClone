# Deploying the Teleop Docker Environment on a New Orin Machine

This guide explains how to transfer and deploy the generated `omniclone-release.tar` image to a new Nvidia Jetson Orin device.

## Prerequisites
- The target Orin machine must have Docker installed.
- Ensure you have transferred the generated `omniclone-release.tar` file to the new Orin machine (e.g., via `scp` or a USB drive).

## Step 1: Load the Docker Image
On the new Orin machine, load the `.tar` archive into your local Docker registry. Navigate to the directory where you placed the `.tar` file and run:

```bash
# Provide permissions if necessary to access docker
sudo docker load -i omniclone-release.tar
```
*Depending on your docker group setup, you may need to prefix commands with `sudo`.*

You can verify the image has been loaded by running:
```bash
docker images
```
You should see `omniclone-release` listed with the `latest` tag.

## Step 2: Running the Interactive Environment
You can start an interactive shell within the container, inheriting the host's networking (which allows you to communicate with ROS, mocap servers, and device ports on the host system):

```bash
# If accessing specific serial devices like robot hands/video senders, you may need to add hardware access
# e.g., --privileged -v /dev:/dev
docker run -it --net=host --ipc=host omniclone-release:latest
```

## Step 3: Running Pre-packaged Tools

The docker image comes pre-configured with several essential tools ready to test.

**1. Test OrinVideoSender:**
```bash
docker run -it --net=host --ipc=host --privileged -v /dev:/dev --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all omniclone-release:latest bash -c 'cd /home/unitree/tools/XRoboToolkit-Orin-Video-Sender && ./open_image_sender.sh'
```

**2. Test Inspire Hand Driver:**
If the hand driver communicates through USB/Serial, make sure to mount your `/dev` devices by appending `-v /dev:/dev --privileged` to your docker command.
```bash
docker run -it --net=host --ipc=host --privileged -v /dev:/dev omniclone-release:latest python /home/unitree/tools/inspire_hand_ws/inspire_hand_sdk/example/Headless_driver_double.py
```

**3. Run ANY2Humanoid Low Command Publisher:**
Open a new terminal and run `lowcmd_publisher.py` inside the offline deep-learning container:
```bash
docker run -it --init --net=host --ipc=host omniclone-release:latest bash -c "source /opt/ros/foxy/setup.bash && source /home/unitree/project/public/unitree_ros2/install/setup.bash && python lowcmd_publisher.py"
```

**4. Run ANY2Humanoid Root Velocity Server:**
Open another terminal and run this *inside* the offline deep-learning container. Since this script heavily relies on PyTorch / CUDA inference, you MUST provide it hardware access to the Tegra GPU components utilizing `--privileged -v /dev:/dev` and properly invoke the NVIDIA runtime stack:
without hand
```bash
docker run -it --init --net=host --ipc=host --privileged -v /dev:/dev --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all omniclone-release:latest bash -c "source /opt/ros/foxy/setup.bash && source /home/unitree/project/public/unitree_ros2/install/setup.bash && python server_raw.py"
```
with hand
```bash
docker run -it --init --net=host --ipc=host --privileged -v /dev:/dev --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all omniclone-release:latest bash -c "source /opt/ros/foxy/setup.bash && source /home/unitree/project/public/unitree_ros2/install/setup.bash && python server_collect.py"
```

## Managing the Docker Container

**1. Check Running Containers (Status):**
To see if your teleop container is currently running, open a new terminal on the host and run:
```bash
docker ps
```
This will list active containers, their Container ID, and their status.

**2. Accessing a Running Container:**
If you already started the container in the background or left it running in another window, you can open a new shell inside that same container using its Container ID or Name:
```bash
docker exec -it <container_id_or_name> bash
```

**3. Check All Containers (Including Stopped Ones):**
```bash
docker ps -a
```

## Additional Notes
- **Networking (`--net=host`):** Using `--net=host` gives the container direct access to the Orin PC's network interfaces. This means:
  - **Low-level control (192.168.123.164):** The container will automatically route ethernet traffic to the low-level motor controller just like the native OS does.
  - **High-level control (192.168.8.105):** The container will seamlessly connect and receive data over the WLAN interface (e.g. from a remote laptop or mocap PC) on the 192.168.8.x subnet without any special port mappings.
- The Docker environment is built exactly from the host `teleop` conda environment without requiring redownloading or internet access.
- Code edits made inside the container are temporary unless you explicitly volume mount your workspace directory (`-v /absolute/path/on/host:/home/unitree/project/ANY2Humanoid`).
