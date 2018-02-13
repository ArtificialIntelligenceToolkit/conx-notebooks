# How To Run Conx

This page describes three categories of running Conx. Conx requires two main components: the Jupyter notebook system, and the Python Conx libraries. The first two options have Jupyter already installed and ready to run. The first option has everything:

* Pre-Built Virtual Machine - download image and run
* In-the-cloud Jupyter - login and install Conx Python packages
* Manage Your Own Install - uses standard Python tools on your computer

There are pros and cons to each option. See the Troublingshooting section at the bottom for additional help.

## Pre-Build Virtual Machines

Rather than installing Conx piecemeal, consider one of the following pre-built virtual machine options:

* [MyBinder](https://mybinder.org/v2/gh/Calysto/conx/master?filepath=binder%2Findex.ipynb) - A complete in-the-cloud option. Availability may be limited due to demand.
* [Conx Docker container](Docker.md) - Perhaps the easiest option. But large download.
* [VirtualBox image](VirtualBox.md) - Perhaps the most flexible option. Complete Ubuntu system, but large download.

## In-the-Cloud Jupyter

These options take care of the Jupyter service, but you still need to install the Python packages. Not all Conx functionality is available.

* [Google's Colabatory](http://colab.research.google.com), [FAQ](https://research.google.com/colaboratory/faq.html)
* [SageMath's Cocalc](http://cocalc.com), [More information](https://cocalc.com/help?session=default)
* [Microsoft's Azure](https://notebooks.azure.com/), [More Information](https://notebooks.azure.com/help)
* [IBM's Cognitive Class](https://datascientistworkbench.com/), [More information](http://support.datascientistworkbench.com/knowledgebase)

You'll need to do the `pip install` commands from the following section.

## Manage Your Own Install

The rest of this document regards managing your own installation of Jupyter and Conx Python libraries.

Ok, this is the hardest option, but perhaps you want to get Conx running on your own computer. It is highly recommended that you use the [Anaconda distribution](https://www.anaconda.com/distribution/).

Conx requires Python3, Keras version 2.0.8 or greater, and some other Python modules that are installed automatically with pip.

On Linux, you may need to install `libffi` and `libffi-dev` in order to render layers for the network display. If you attempt to display a network and it appears empty, or if you attempt to network.propagate_to_image() and it gives a PIL error, you need these libraries. On Ubuntu or other Debian-based system you can install them with:

```bash
sudo apt install libffi-dev libffi6
```
Next, we use `pip` to install the Python packages. We use `pip` rather than `conda` because conx is not yet available as a conda package.

**Note**: you may need to use `pip3`, or admin privileges (eg, sudo), or install into a user environment.

```bash
pip install conx -U --user
```

You will need to decide whether to use Theano, TensorFlow, or CNTK. Pick one. See [docs.microsoft.com](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) for installing CNTK on Windows or Linux. All platforms can also install either of the others using pip:

It is recommended that you use TensorFlow. Do one of the following:

```bash
pip install tensorflow
```

**or**

```bash
pip install theano
```

**or**

Install the Cognitive Toolkit:

https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine

### Operating-Specific Details

On MacOS, you may also need to render the SVG visualizations:

```bash
brew install cairo
```

To make MP4 movies, you will need the `ffmpeg` executable installed and available on your default path.

On MacOS, you could use:

```bash
brew install ffmpeg
```

On Windows:

See, for example, https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg

On Linux:

```bash
sudo apt install ffmpeg
# or perhaps:
sudo yum install ffmpeg
```

### Use with Jupyter Notebooks

To use the Network.dashboard() and camera functions, you will need to enable `ipywidgets`:

``` bash
jupyter nbextension enable --py widgetsnbextension
```

If you install via conda, then it will already be enabled:

``` bash
conda install -c conda-forge ipywidgets
```

### Setting the Keras Backend

To use a Keras backend other than TensorFlow, edit (or create) `~/.keras/kerson.json`, like:

```json
{
    "backend": "theano",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

### Troubleshooting

1. If you have a problem after installing matplotlib with pip, and you already have matplotlib installed (say, with apt) you may want to remove the apt-installed version of matplotlib.
2. Theano has many known problems. Don't use Theano, use TensorFlow.

If you have additional problems, please see the con-users mailing list:

https://groups.google.com/forum/#!forum/conx-users
