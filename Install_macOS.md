## Prerequisites for Matterport3D Simulator on MacOS

A C++ compiler with C++11 support is required. Matterport3D Simulator has several dependencies:
- [OpenCV](http://opencv.org/) >= 2.4 including 3.x 
- [OpenGL](https://www.opengl.org/)
- [OSMesa](https://www.mesa3d.org/osmesa.html)
- [GLM](https://glm.g-truc.net/0.9.8/index.html)
- [Numpy](http://www.numpy.org/)
- [pybind11](https://github.com/pybind/pybind11) for Python bindings
- [Doxygen](http://www.doxygen.org) for building documentation

### Install dependencies on MacOS
Install dependencies on Ubuntu is easy and straightforward, but it gets a little bit tricky on MacOS.

FIrst let's install [Conda](https://www.anaconda.com/download/).

Install [Homebew](https://brew.sh/). We will be using it to install other dependencies.

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

#### Install Opencv via Homebew.

```bash
brew install opencv3 --with-python3
```
Note that this will install OpenCV without OpenGL support. This will be enough if you don't need it with the Matterport3D Simulator, i.e. 
```python
sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
```

After the above step, your OpenCV is installed, but we would want it to be used in a controllable environment.

Create Anaconda enviorment with python 3.6 version. 
(We will probably be using AllenNLP, and it currently supports only python 3.6)

```bash
# Choose whichever names of the environment you like, I am using 'r2r36' here.
conda create -n r2r36 python=3.6
```

```bash
# activate the env 
source activate r2r36

# install numpy. It will later on be used by OpenCV
pip install numpy
```

Now that we have OpenCV installed and conda env set up, we need to sym-link the OpenCV bindings.

Go to the site-package in your env
```bash
cd /anaconda3/envs/r2r36/lib/python3.6/site-packages

# create sym-link to where you installed OpenCV
ln -s /usr/local/opt/opencv/lib/python3.6/site-packages/cv2.cpython-36m-darwin.so cv2.so
```

Test if the OpenCV installation is correct. You should something similar like this:
```
(r2r36) cma@cma-ltm:/anaconda3/envs/r2r36$ python 
Python 3.6.5 |Anaconda, Inc.| 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> 
```

#### Install OpenCV from source within Anaconda Environment
The following instruction will guide you to install OpenCV 3.4.1.

As stated earlier, we will not use the OpenCV provided in Anaconda, because we need OpenGL support for rendering in Matterport3D Simulator.

```bash
# Download the OpenCV from their GitHub repo
cd ~
git clone https://github.com/opencv/opencv
cd opencv
git checkout 3.4.1

mkdir release
cd release

# activate the enviorment, if you haven't
source activate r2r36

# configure the OpenCV with your conda enviorment
cmake .. \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_OPENGL=ON \
    -DWITH_OPENVX=ON \
    -DWITH_QT=ON \
    -DWITH_OPENCL=ON \
    -DBUILD_PNG=ON \
    -DBUILD_TIFF=ON \
    -DWITH_1394=OFF \
    -DWITH_CUDA=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=`which python3` \
    -DPYTHON3_LIBRARY=`python3-config --configdir`/libpython3.6.dylib \
    -DPYTHON3_INCLUDE_DIR=`python3 -c "import distutils.sysconfig as s; print(s.get_python_inc())"`
```
Compile and Install OpenCV source
```bash
# compile with 8 threads
make -j8
make install   # do not use sudo!
```

Now that we have OpenCV installed and conda env set up, we need to sym-link the OpenCV bindings.

Go to the site-package in your env
```bash
cd /anaconda3/envs/r2r36/lib/python3.6/site-packages

# create sym-link to where you installed OpenCV
ln -s ~/opencv/release/lib/python3/cv2.cpython-36m-darwin.so cv2.so
```

Test if the OpenCV installation is correct. You should something similar like this:
```
(r2r36) cma@cma-ltm:/anaconda3/envs/r2r36$ python 
Python 3.6.5 |Anaconda, Inc.| 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> 
```


#### Install OpenGL
Make sure you are not in the virtual env. 
```bash
source deactivate

```
We need to install GLFW3 and Glew
```bash
brew install glfw3 glew
```

#### Install OSMesa
In MacOS, OSMesa will be installed by installing [XQuartz](https://www.xquartz.org/). You can dowload it directly from the URL.

You should be prompted to log out and log in again.

#### Install GLM, Pybind11, and Doxygen
```bash
brew install glm pybind11 doxygen
```

## Compiling

Before compiling, you may want to make sure you have all the compiling tools you need, e.g. cmake, pkg-config, etc.
```bash
brew install cmake pkg-config jsoncpp
```


Build OpenGL version using CMake:
```bash
# Make sure you are in the working directory for Matterport3DSimulator
cd ~/Documents/Matterport3DSimulator

# create the folder
mkdir build_mac && cd build_mac

# activate the env you created earlier with conda
source activate r2r36

# if you use conda, make sure it finds the correct python path to your virtual env
cmake -DPYTHON_EXECUTABLE:FILEPATH=/path/to/your/bin/python ..
make
```

Check if the installation of MatterSim is correct. You should see something similar like below. 
```
(r2r36) cma@cma-ltm:~/Documents/Matterport3DSimulator/build_mac$ python 
Python 3.6.5 |Anaconda, Inc.| 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import MatterSim
>>> 
```