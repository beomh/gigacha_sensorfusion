docker run -it \
-v $(pwd)/workspace:/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=unix$DISPLAY \
--net=host \
--privileged \
lidar_to_camera_projection 
