# How To Run ConX

This page describes FOUR options for running ConX. ConX requires two main components: the Jupyter notebook system, and the Python ConX libraries. The first three options have Jupyter already installed and ready to run. The first option has everything:

* Free, Online, Ready-to-run - click and go
* Pre-Built Virtual Machine - download image and run
* In-the-cloud Jupyter - login and install ConX Python packages
* Manage Your Own Install - uses standard Python tools on your computer

There are pros and cons to each option. See the Troubleshooting section at the bottom for additional help.

## Free, Online, Ready-to-run

* [MyBinder](https://mybinder.org/v2/gh/Calysto/conx/master?filepath=binder%2Findex.ipynb) - A complete in-the-cloud option. Availability may be limited due to demand. Just click the link, and you'll find yourself running a notebook ready to build neural networks. Weird, I know, right?

## Pre-Built Virtual Machines

Rather than installing ConX piecemeal, consider one of the following pre-built virtual machine options:

* [ConX Docker container](Docker.md) - Perhaps the easiest option. But large download.
* [VirtualBox image](VirtualBox.md) - Perhaps the most flexible option. Complete Ubuntu system, but large download.

## In-the-Cloud Jupyter

These options take care of the Jupyter service, but you still need to install the Python packages. Not all ConX functionality is available on all of these options.

* [Google's Colabatory](http://colab.research.google.com), [FAQ](https://research.google.com/colaboratory/faq.html)
* [SageMath's Cocalc](http://cocalc.com), [More information](https://cocalc.com/help?session=default)
* [Microsoft's Azure](https://notebooks.azure.com/), [More Information](https://notebooks.azure.com/help)
* [IBM's Cognitive Class](https://datascientistworkbench.com/), [More information](http://support.datascientistworkbench.com/knowledgebase)

You'll need to do the `pip install` commands from the following section. However, services like Colab and Cocalc allow the sharing of notebooks with others. Some allow persistent storage of your notebooks. Google's Colab offers runtimes with GPU support.

## Manage Your Own Install

The rest of this document explains how to manage your own installation of Jupyter and ConX Python libraries.

Ok, this is the hardest option, but perhaps you want to get ConX running on your own computer. It is highly recommended that you use the [Anaconda distribution](https://www.anaconda.com/distribution/).

ConX requires Python3, Keras version 2.0.8 or greater, and some other Python modules that are installed automatically with pip.

On Linux, you may need to install `libffi` and `libffi-dev` in order to render layers for the network display. If you attempt to display a network and it appears empty, or if you attempt to network.propagate_to_image() and it gives a PIL error, you need these libraries. On Ubuntu or other Debian-based system you can install them with:

```bash
sudo apt install libffi-dev libffi6
```
Next, we use `pip` to install the Python packages. We use `pip` rather than `conda` because conX is not yet available as a conda package.

**Note**: you may need to use `pip3`, or admin privileges (eg, sudo), or install into a user environment.

```bash
pip install conx -U --user
```

You will need to decide whether to use TensorFlow, Theano, or CNTK. Pick one. See [docs.microsoft.com](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) for installing CNTK on Windows or Linux. All platforms can also install either of the others using pip:

It is strongly recommended that you use TensorFlow. Do one of the following:

```bash
pip install tensorflow
```

**or**

Install the Cognitive Toolkit:

https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine

### Operating-Specific Details

On MacOS, you may want to render the SVG visualizations. This feature requires the cairo library:

```bash
brew install cairo
```

To make MP4 movies, you will need the `ffmpeg` executable installed and available on your default path.

On MacOS, you could use:

```bash
brew install ffmpeg
```

On Windows:

To install cairocffi, if you are using conda, try:

```
activate your-python-environment
conda install -f -c rmg cairocffi
```

To install ffmpeg, if you are using conda, try:

```
conda install -c conda-forge ffmpeg 
```

Otherwise, see https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg

On Linux:

```bash
sudo apt install ffmpeg
# or perhaps:
sudo yum install ffmpeg
```

### Use with Jupyter Notebooks

To use the Network.dashboard() and camera functions, you will need to enable `ipywidgets`. If you install via conda, then it will already be enabled:

``` bash
conda install -c conda-forge ipywidgets
```

If you use pip, you will need to enable `ipywidgets`:

``` bash
jupyter nbextension enable --py widgetsnbextension
```

### Setting the Keras Backend

To use a Keras backend other than TensorFlow, edit (or create) `~/.keras/keras.json`, like:

```json
{
    "backend": "tensorflow",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

### Troubleshooting

1. If you have a problem after installing matplotlib with pip, and you already have matplotlib installed (say, with apt) try removing the apt-installed version of matplotlib, or the pip install matplotlib. You may have to do this manually if pip no longer works.
2. Theano has many known problems. Don't use Theano, use TensorFlow.

If you have additional problems, please see the conx-users mailing list:

https://groups.google.com/forum/#!forum/conx-users
