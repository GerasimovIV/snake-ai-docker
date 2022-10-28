docker run -it -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/snd --gpus all --name snake_ai_container snake_ai
