# snake-ai-docker

#Install

```bash
git clone https://github.com/GerasimovIV/snake-ai-docker.git
cd snake-ai-docker
```

```
sh docker_build.sh
```

```
xhost +
sh docker_run.sh
```

# Inside Docker container

 ```bash
 cd snake-ai-pytorch/
 
 export QT_X11_NO_MITSHM=1
 
 python3 agent.py
 
 ```
