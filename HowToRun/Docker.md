# Running Conx in Docker

These are the steps required to download and run Conx in a Docker container.

This option will allow you to run all of Conx via a webbrowser on your operating system. It installs a full Ubuntu system, but only runs a webserver from it.

* Docker Program size: 300 MB
* Docker Container size: 7 GB

1. **Download Docker**. You will want to download the Docker Community Edition (CE):

https://www.docker.com/community-edition

On Mac and Windows, you will need to have Docker running in order to do the following steps. On Linux, docker will run as needed.

2. **Download Container**. Download the calysto/conx Docker container by either entering `calysto/conx` in the Docker program, or from the command line:

```
sudo docker pull calysto/conx
```

3. **Run Container**. You can then run the container by typing the following command in the terminal. You should enter it exactly as shown:

```
sudo docker run --rm -v "$PWD":/home/jovyan/work -p 8888:8888 calysto/conx
```

4.**Access the Webbrowser**.  Open the URL shown at the bottom of the message from the above command.

5. **Test Conx**. In the upper righthand corner of the browser, choose New -> Python 3 to start a new Python 3 notebook running. Then type:

```
import conx as cx
```

and press Shift-RETURN. You should see the message:

```
Using TensorFlow backend.
Conx, version 3.6.0
```

6. **Shutdown**. If this works, you're done! You can shut down the notebook by
select `File` -> `Close and Halt`. You can shutdown the Docker container and webserver
by typing Control+C in the console where you entered `sudo docker run...` in step #2.
If this doesn't work, then you want to try the Docker option,
or ask from help on the `conx-users` mailing list:

https://groups.google.com/forum/#!forum/conx-users


7. **Regular Use**. To use the Docker container regularly, go to step #3.
