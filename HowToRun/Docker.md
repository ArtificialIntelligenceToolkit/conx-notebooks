# Running Conx in Docker

These are the steps required to download and run Conx in a Docker container.

* Docker Program size: 
* Docker Container size: 

1. Download Docker. You will want to download the Docker Community Edition (CE):

https://www.docker.com/community-edition

2. Download the calysto/conx Docker container:

```
sudo docker pull calysto/conx
```

3. Run the image:

```
sudo docker run --rm -v "$PWD":/home/jovyan/work -p 8888:8888 calysto/conx
```

4. Open the URL shown at the bottom of the message from the above command.

5. In the upper right corner of the browser, choose New -> Python 3 to start a
new Python 3 notebook running. Then type:

```
import conx as cx
```

and press Shift-RETURN. You should see the message:

```
Using TensorFlow backend.
Conx, version 3.6.0
```

6. If this works, you're done! You can shut down the notebook by
select `File` -> `Close and Halt`
