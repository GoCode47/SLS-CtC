#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
//#include <cv.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <conio.h>
//#include<array>

using namespace cv;
using namespace std;

double euclidean_distance(Point2f a, Point2f b) {
	return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

double angle(Point2f a, Point2f b) {
	return atan((a.y - b.y) / (a.x - b.x));
}

Mat pre_process(Mat src) {
	Mat1b mask1, mask2, mask, src2;
	Mat grayImg, blurred, otsu, canny, hsv;

	GaussianBlur(src, src2, Size(3, 3), 0, 0);

	cvtColor(src2, hsv, COLOR_BGR2HSV);

	vector<Mat> channels;
	//GaussianBlur(hsv, hsv2, Size(9, 9), 0, 0);
	//bilateralFilter(hsv, hsv2, 5, 70, 70);
	split(hsv, channels);

	float red1 = 25 / 2.0f;
	float red2 = (360 - 25) / 2.0f;

	mask1 = channels[0] < red1;
	mask2 = channels[0] > red2;

	mask = (mask1 | mask2);
	mask = ~mask;
	//imshow("Mask", mask);

	if (src.channels() == 3)
		cvtColor(src, grayImg, CV_BGR2GRAY);
	else if (src.channels() == 4)
		cvtColor(src, grayImg, CV_BGRA2GRAY);
	else grayImg = src;

	//equalizeHist(grayImg, grayImg);
	double o = threshold(grayImg, otsu, 0, 255, CV_THRESH_BINARY | CV_ADAPTIVE_THRESH_GAUSSIAN_C);
	otsu = ~otsu;
	multiply(mask, otsu, otsu);
	imshow("Out", otsu);

	Canny(otsu, canny, o, o * 1 / 2, 3, 1);
	imshow("Canny", canny);
	return canny;
}

Mat CtC_features(Mat canny)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int start = clock();

	/// Find contours
	findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	///Get the mass centers
	Point2f centroid;
	double xmid = 0, ymid = 0;
	long long n_points = 0;
	vector<Point2f> mc(contours.size());
	for (int i = 0; i<contours.size(); i++) {
		n_points += contours[i].size();
		for (int j = 0; j < contours[i].size(); j++) {
			Point2f point = contours[i][j];
			xmid += point.x;
			ymid += point.y;
		}
	}
	centroid.x = xmid / n_points;
	centroid.y = ymid / n_points;
	//cout << centroid.x << " " << centroid.y<<endl;

	vector<pair<double, double>> feature_v;
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours[i].size(); j++) {
			Point2f curr = contours[i][j];
			double dist = euclidean_distance(centroid, curr);
			double ang = angle(centroid, curr);
			if (centroid.x > curr.x) ang += 3.14159;  //PI
													  //cout << dist << " " << ang << endl;
			feature_v.push_back(make_pair(ang, dist));
		}
	}
	sort(feature_v.begin(), feature_v.end());

	int degree = 5;
	vector<double> temp, feature(360 / degree, 0);
	double interval = double((double(5) / double(360)) * 2 * 3.14159); //5 degrees interval
																	   //cout << interval << endl;
	double ang = -1.57079;
	int j = 0;
	for (int i = 0; i < feature_v.size(); i++) {
		while (feature_v[i].first > ang) {
			//cout << ang << endl;
			ang += interval;
			if (temp.empty()) temp.push_back(0);
			feature[j++] = *max_element(temp.begin(), temp.end());
			temp.clear();
		}
		temp.push_back(feature_v[i].second);
	}

	double maxf = *max_element(feature.begin(), feature.end());
	for (int i = 0; i < 360 / degree; i++)
		feature[i] /= maxf;

	Mat f = Mat(feature).reshape(0, 1);
	f.convertTo(f, CV_32F);

	int end = clock();
	//cout<< "Execution time : " << (end - start) / double(CLOCKS_PER_SEC) * 1000 << " ms"<< endl;
	/*for (int i = 0; i < feature.size(); i++)
	cout << feature[i] << endl;
	cout << feature.size() << endl;  */

	/// Draw contours
	/*Mat drawing = Mat::zeros(canny.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing); */

	return f;
}

int main(int argc, char** argv) {
	int file_num = 3168;
	int startnum = 2500;

	double scaleStep = 1.1;
	Size minimalObjectSize(1, 1);
	Size maximalObjectSize(70, 70);

	std::stringstream inp_path30;
	inp_path30 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/30.jpg";
	Mat img30 = imread(inp_path30.str());

	std::stringstream inp_path40;
	inp_path40 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/40.jpg";
	Mat img40 = imread(inp_path40.str());

	std::stringstream inp_path50;
	inp_path50 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/50.jpg";
	Mat img50 = imread(inp_path50.str());

	std::stringstream inp_path60;
	inp_path60 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/60.jpg";
	Mat img60 = imread(inp_path60.str());

	std::stringstream inp_path70;
	inp_path70 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/70.jpg";
	Mat img70 = imread(inp_path70.str());

	std::stringstream inp_path80;
	inp_path80 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/80.jpg";
	Mat img80 = imread(inp_path80.str());

	std::stringstream inp_path90;
	inp_path90 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/90.jpg";
	Mat img90 = imread(inp_path90.str());

	std::stringstream inp_path100;
	inp_path100 << "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/100.jpg";
	Mat img100 = imread(inp_path100.str());

	CascadeClassifier detector;
	string cascadeName = "C:/Users/L3IN/Documents/Visual Studio 2015/Projects/SLS-CtC/traffic_sign.xml";
	bool loaded = detector.load(cascadeName);
	double groundThreshold = 1;

	//cout << "Starting\n";
	int j = 1;
	for (int num = 1; num <= 3168; num++) {
		cout << num << endl;
		Mat img, img_re, img_show;
		std::stringstream inp_path;
		inp_path << "C:/Users/L3IN/Downloads/vi_speed_limit/image (" << num << ").jpg";
		img = imread(inp_path.str());
		if (img.empty()) continue;
		img_show = imread(inp_path.str());
		clock_t start = clock();
		Mat hsv, total_re, yuv;
		vector<Mat> channels, channels_yuv;
		split(img, channels);
		channels[0] += 150;
		channels[1] -= 30;
		channels[2] += 150;
		merge(channels, img);
		Mat red = channels[2];
		cvtColor(img, yuv, CV_BGR2YUV);
		split(yuv, channels_yuv);
		Mat yuv_ch1 = channels_yuv[1];
		equalizeHist(yuv_ch1, yuv_ch1);


		vector<Rect> found;
		found.clear();

		detector.detectMultiScale(yuv_ch1, found, scaleStep, groundThreshold, 0 | 2, minimalObjectSize, maximalObjectSize);


		IplImage* image_target = new IplImage(img_show);

		if (found.size() > 0) {
			for (int i = 0; i <= found.size() - 1; i++) {
				for (int i = 0; i <= found.size() - 1; i++) {
					rectangle(img_show, found[i].br(), found[i].tl(), Scalar(0, 255, 0), 3, 8, 0);
				}
				cvSetImageROI(image_target, found[i]);
			}

			Mat target = cvarrToMat(image_target), target_re;

			Size size1(84, 96);
			resize(target, target_re, size1);
			//imshow("temp", target_re);
			/*Mat bgr = target_re;

			Mat grayImg, blur, otsu, canny, temp, cropped;
			if (target_re.channels() == 3)
				cvtColor(target_re, grayImg, CV_BGR2GRAY);
			else if (target_re.channels() == 4)
				cvtColor(target_re, grayImg, CV_BGRA2GRAY);
			else grayImg = target_re;

			double o = threshold(grayImg, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			otsu = ~otsu;
			imshow("otsu", otsu);
			cvWaitKey(1);
			Canny(otsu, temp, o, o * 1 / 2, 3, 1);

			vector<Vec3f> circles;
			HoughCircles(temp, circles, CV_HOUGH_GRADIENT, 2, 50, 200, 60, 30, 0);

			if (circles.empty()) continue;

			Rect roi(int(circles[0][0] - 0.72*circles[0][2]), int(circles[0][1] - 0.72*circles[0][2]), int(1.41*circles[0][2]), int(1.41*circles[0][2]));
			cropped = bgr(roi);

			imshow("cropped", cropped);*/
			IplImage* number = new IplImage(target_re);
			cvSetImageROI(number, cvRect(20, 25, 50, 45));
			Mat number_mat = cvarrToMat(number), number_gr, number_bi;
			if (!number_mat.empty()) {
				ostringstream name;
				name << "C:/Users/L3IN/Downloads/image data set/out3/res" << j << ".bmp";
				imwrite(name.str(), number_mat);
				j++;
			//cout << j++ << endl;
			}
			//imshow("target", number_mat);

			/*clock_t cl_start = clock();

			Mat processed_image = pre_process(cropped);
			Mat feature_vector = CtC_features(processed_image);
			//feature_vector.convertTo(feature_vector, CV_32F);
			//cout << feature_vector.rows << " " << feature_vector.cols << " " <<feature_vector.type()<< endl;

			Ptr<ml::SVM> svm = ml::SVM::create();
			svm = ml::SVM::load("SVM.xml");
			int label = svm->predict(feature_vector);
			cout << label << endl;

			if (label == 1) {
				cout << "Speed limit = 30 km/h   ";
				imshow("result_sign", img30);
			}
			if (label == 2) {
				cout << "Speed limit = 40 km/h   ";
				imshow("result_sign", img40);
			}
			if (label == 3) {
				cout << "Speed limit = 50 km/h   ";
				imshow("result_sign", img50);
			}
			if (label == 4) {
				cout << "Speed limit = 60 km/h   ";
				imshow("result_sign", img60);
			}
			if (label == 5){
				cout << "Speed limit = 80 km/h   ";
				imshow("result_sign", img80);
			}
			if (label == 6) {
				cout << "Speed limit = 100 km/h   ";
				imshow("result_sign", img100);
			}
			if (label == 7) {
				cvWaitKey(1);
				continue;
			}

			clock_t cl_end = clock();

			imshow("result", img_show);
			clock_t end = clock();
			double work_time = double(end - start) / CLOCKS_PER_SEC;
			int classification_time = (double(cl_end - cl_start) / CLOCKS_PER_SEC) * 1000;
			cout << "work_time = " << work_time << " s   classification time = " << classification_time << " ms   frame_num = " << num << endl; */

			cvWaitKey(1);
		}
	}
	_getch();
	return 0;
}
