# gun-detection
## Local Installation from Source
### Install Dependencies:
* <code>sudo apt-get install build-essential git cmake libopencv-dev libconfig++-dev libspeechd-dev</code>
* <code>git clone https://github.com/AlexeyAB/darknet.git</code>
* <code>cd darknet</code>
* <code>vi makefile #set LIBSO=1, and other desired flags such as GPU=1, CUDNN=1, CUDNN_HALF=1, OPENCV=1, and uncomment the ARCH= line that corresponds to your GPU</code>
* <code>make</code>
* <code>sudo cp libdarknet.so /usr/local/lib/</code>
* <code>sudo cp include/darknet.h /usr/local/include/</code>
* <code>sudo ldconfig</code>
* Download the DarkHelp Library from here: <link>https://www.ccoderun.ca/download/</link> (Tested on darkmark-1.0.0-2968-Linux.deb) and run <code>sudo dpkg -i darkhelp-*.deb</code>
### Compile:
Run <code>cmake .</code> and <code>make</code> in the repo root directory.
## Demo
Download Pre-trained model: <code>./download_weights.sh</code>
### Configuration
Open default_config.cfg and change the "sources" parameter in the format of <code>["Name","Path to source"]</code> for each desired video source. Can read in sources from an IP camera in the following example format:

<code>"http://192.168.0.80:8080/video?dummy=param.mjpg"</code>

Update the "num_windows" parameter to reflect the number of sources you added and ensure that "num_columns" times "num_rows" is greater than or equal to "num_windows".

### Run
Run <code>./live_demo default_config.cfg</code>

If "webserver_flag" is set to true the output will be streamed to localhost:8888 by default (the port can be changed with the "webserver_port" paramater) and can be accessed on the LAN through http://ip-address:8888 where ip-address can be found by running <code>hostname -I</code>
