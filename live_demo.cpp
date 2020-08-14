#include <DarkHelp.hpp>
#include <mutex>
#include <thread>
#include "MJPEGWriter.h"
#include <libconfig.h++>
#include "date.h"
#include <libspeechd.h>

using namespace cv;
using namespace libconfig;
using namespace std;

int display_width;
int display_height;
Mat noInput;
vector<vector<queue<Mat>>> buffer;
vector<vector<Mat>> lFrame;
vector<vector<bool>> done;
int doneCount = 0;
mutex mtx;
mutex mtx1;

void cache(int x, int y, string vidSourcePath, int duration){
	VideoCapture cacheCap(vidSourcePath);
	while (true){
		Mat frame;
		cacheCap >> frame;
		if (frame.empty()){
			lFrame[x][y]=noInput;
			done[x][y]=1;
			mtx1.lock();
			doneCount++;
			mtx1.unlock();
			break;
		}
		mtx.lock();
		lFrame[x][y] = frame;
		double fps = cacheCap.get(CAP_PROP_FPS);
		buffer[x][y].push(frame);
		if(buffer[x][y].size() >= duration*fps){
			buffer[x][y].pop();
		}
		mtx.unlock();
		frame.release();
	}
	cacheCap.release();
}

const string currentDateTime() {
	using namespace date;
    auto now = std::chrono::system_clock::now();
	stringstream buffer;
	buffer << now;
	return buffer.str();
}

void vid_record (int x, int y, queue<Mat> tempBuffer, string path, string vidSourceName, string time, int fps){
	string name = path+vidSourceName+":"+time+".avi";
	VideoWriter video(name,VideoWriter::fourcc('M','J','P','G'),fps, Size(display_width, display_height));
	while (tempBuffer.empty() == false){
		video.write(tempBuffer.front());
		tempBuffer.pop();
	}
	video.release();
}

int main(int argc, char* argv[]){
	Config cfg;
	const char * configFile = argv[1];
	try
	{
		cfg.readFile(configFile);
	}
	catch(const FileIOException &fioex)
	{
		cerr << "I/O error while reading file." << endl;
		return(EXIT_FAILURE);
	}
	catch(const ParseException &pex){
		cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << endl;
		return(EXIT_FAILURE);
	}
	bool webserver_flag = cfg.lookup("webserver_flag");
	bool local_flag = cfg.lookup("local_flag");
	bool display_detection_only_flag = cfg.lookup("display_detection_only_flag");
	bool vid_record_flag = cfg.lookup("vid_record_flag");
	bool img_record_flag = cfg.lookup("img_record_flag");
	bool vid_yellow_record_flag = cfg.lookup("vid_yellow_record_flag");
	bool img_yellow_record_flag = cfg.lookup("img_yellow_record_flag");
	bool auditory_cue_flag = cfg.lookup("auditory_cue_flag");
	int webserver_port = cfg.lookup("webserver_port");
	string darknet_cfg = cfg.lookup("darknet_cfg");
	string darknet_weights = cfg.lookup("darknet_weights");
	string darknet_classes = cfg.lookup("darknet_classes");
	int num_windows = cfg.lookup("num_windows");
	int num_columns = cfg.lookup("num_columns");
	int num_rows = cfg.lookup("num_rows");
	if (num_windows > num_columns*num_rows){
		cout << "Error: Too many windows for given columns and rows." << endl;
		return(EXIT_FAILURE);
	}
	vector<vector<vector<string>>> vidSource(num_columns, vector<vector<string>>(num_rows, {"",""}));
	const Setting& sources = cfg.lookup("sources");
	int count = 0;
	for(int i = 0; i < num_columns; i++){
		for(int j = 0; j < num_rows; j++){
			if(count<num_windows){
				const Setting & names = sources[count];
				string temp = names[0];
				vidSource[i][j][0] = temp;
				string clear = names[1];
				vidSource[i][j][1] = clear;
			}
			count++;
		}
	}
	display_width = cfg.lookup("display_width");
	display_height = cfg.lookup("display_height");
	string img_record_path = cfg.lookup("img_record_path");
	string vid_record_path = cfg.lookup("vid_record_path");
	int vid_record_duration = cfg.lookup("vid_record_duration");
	int vid_record_playback_fps = cfg.lookup("vid_record_playback_fps");
	double yellow_alert_threshold = cfg.lookup("yellow_alert_threshold");
	int red_alert_check_frames = cfg.lookup("red_alert_check_frames");
	int red_alert_min_detect = cfg.lookup("red_alert_min_detect");
	Mat temp(display_height, display_width, CV_8UC3, Scalar(255,0,255));
	putText(temp, "NO SOURCE", Point(10,125),FONT_HERSHEY_COMPLEX, 5, Scalar(255,255,255), 10, LINE_AA);
	temp.copyTo(noInput);
	temp.deallocate();
	queue<Mat> empty;
	buffer.resize(num_columns, vector<queue<Mat>>(num_rows, empty));
	lFrame.resize(num_columns, vector<Mat>(num_rows, noInput));
	done.resize(num_columns, vector<bool>(num_rows, false));
    DarkHelp darkhelp(darknet_cfg, darknet_weights, darknet_classes);
	darkhelp.annotation_colours = {{0, 255, 255}};
	darkhelp.annotation_include_duration= false;
	thread caches [num_columns][num_rows];
	namedWindow("Output", WINDOW_NORMAL);
	resizeWindow("Output", display_width, display_height);
	setWindowProperty("Output", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
	count = 0;
	for(int i=0; i<num_columns; i++){
		for(int j=0; j<num_rows; j++){
			if(count<num_windows){
				caches[i][j]=thread(cache, i, j,vidSource[i][j][1], vid_record_duration);
			}
			else
				done[i][j] = 1;
			count++;
		}
	}
	MJPEGWriter webserver(webserver_port);
	if(webserver_flag){
		webserver.write(noInput);
		webserver.start();
	}
	SPDConnection* synth;
	if(auditory_cue_flag)
		synth = spd_open("bash", "main", NULL, SPD_MODE_THREADED);
	thread takeVideo [num_columns][num_rows];
	int frameCount [num_columns][num_rows] = {0};
	int runningCount [num_columns][num_rows] = {0};
	queue<bool> lastFrames [num_columns][num_rows];
	Mat clear(display_height, display_width, CV_8UC3, Scalar(0,255,0));
	putText(clear, "ALL CLEAR", Point(10,125),FONT_HERSHEY_COMPLEX, 5, Scalar(0,0,0), 10, LINE_AA);
	vector<vector<Mat>> lastDetection (num_columns, vector<Mat>(num_rows, clear));
	clear.deallocate();
	while (doneCount<num_windows)
	{
		Mat totalCombined;
		int last = 0;
		for (int i=0; i<num_columns; i++){
			Mat verCombined;
			for (int j=0; j<num_rows; j++){
				Mat output;
				if(done[i][j]==0){
					string time = currentDateTime();
					mtx.lock();
					Mat curr = lFrame[i][j];
					queue<Mat> tempBuffer = buffer[i][j];
					mtx.unlock();
					const auto result = darkhelp.predict(curr, yellow_alert_threshold);
					bool detected = !result.empty();
					if (detected){
						runningCount[i][j]++;
						lastFrames[i][j].push(true);
					}
					else
						lastFrames[i][j].push(false);
					if(lastFrames[i][j].size()==red_alert_check_frames){
						if(lastFrames[i][j].front()==1)
							runningCount[i][j]--;
						lastFrames[i][j].pop();
					}
					bool red_alert = runningCount[i][j]>=red_alert_min_detect&&detected;
					double fps = 1/((double)chrono::duration_cast<std::chrono::milliseconds>(darkhelp.duration).count()/1000)/num_windows;
					string vidSourceName = vidSource[i][j][0];
					if (frameCount[i][j]>=fps*vid_record_duration){
						if((vid_record_flag&&red_alert)||(vid_yellow_record_flag&&detected)){
							takeVideo[i][j]=thread(vid_record, i, j, tempBuffer, vid_record_path, vidSourceName, time, vid_record_playback_fps);
							if(takeVideo[i][j].joinable()){
								takeVideo[i][j].detach();
							}
							frameCount[i][j]=0;
						}
						if (auditory_cue_flag&&red_alert){
							spd_cancel(synth);
							spd_say(synth, SPD_IMPORTANT, vidSource[i][j][0].c_str());
							frameCount[i][j]=0;
						}
					}
					if(red_alert){
						darkhelp.annotation_colours = {{0, 0, 255}};
					}
					if(display_detection_only_flag)
						darkhelp.annotation_include_timestamp = true;
					output = darkhelp.annotate(yellow_alert_threshold);
					if((img_record_flag&&red_alert)||(detected&&img_yellow_record_flag)){
						imwrite( img_record_path+vidSource[i][j][0]+":"+time+".jpg", output );
					}
					string text = vidSource[i][j][0];
					Size textSize=getTextSize(text, FONT_HERSHEY_COMPLEX, 1,1,{0});
					int offset_x = 5;
					int offset_y = 5;
					rectangle(output, Point(offset_x, offset_y), Point(offset_x + 4 + textSize.width, offset_y + 8 +textSize.height),Scalar(255,255,255), FILLED,0,0);
					putText(output, text, Point(offset_x + 2, offset_y +textSize.height+2), FONT_HERSHEY_COMPLEX, 1, Scalar(0,0,0), 1, LINE_AA);
					darkhelp.annotation_colours = {{0, 255, 255}};
					darkhelp.annotation_include_timestamp = false;
					if(display_detection_only_flag){
						if(detected){
							lastDetection[i][j] = output;
						}
						output = lastDetection[i][j];
					}
					frameCount[i][j]++;
					if(detected){
						if(red_alert)
							rectangle(output, Point(0, 0), Point(display_width-1, display_height),Scalar(0,0,255), 5,8,0);
						else
							rectangle(output, Point(0, 0), Point(display_width-1, display_height),Scalar(0,255,255), 5,8,0);						
					}
				}
				else
					output=noInput;
				if(verCombined.empty())
					verCombined = output;
				else
					vconcat(verCombined, output, verCombined);
			}
			if(totalCombined.empty())
				totalCombined = verCombined;
			else
				hconcat(totalCombined, verCombined, totalCombined);
		}
		if(webserver_flag)
			webserver.write(totalCombined);
		if(local_flag){
			imshow("Output", totalCombined);
			if (waitKey(30)>=0)
				break;
		}
	}
	if(auditory_cue_flag)
		spd_close(synth);
	if(webserver_flag)
		webserver.stop();
    return 0;
}
