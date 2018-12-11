#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include <limits.h>

using cv::cvtColor;
using cv::imread;
using cv::imshow;
using cv::Mat;
using cv::Scalar;
using cv::Vec3b;
using cv::Vec4i;
using cv::waitKey;

using std::cout;
using std::string;
using std::vector;

void cropRowDetec(Mat m);
void grayTransform(Mat &m);
void skeletonize(Mat &m);
void houghTransform(Mat &m);
void filterSimLines(vector<Vec4i> &lines);
double solveforB(const Vec4i & twoPts);

const int HOUGH_RHO = 2;					  //Distance resolution of the accumulator in pixels
const double HOUGH_ANGLE = CV_PI / 45.0; //Angle resolution of the accumulator in radians --------- 4 degrees
const int HOUGH_THRESH_MAX = 100;			  //Accumulator threshold parameter.Only those lines are returned that get enough votes
const int HOUGH_THRESH_MIN = 10;
const int HOUGH_THRESH_INCR = 1;
const int NUMBER_OF_ROWS = 5;						 //how many crop rows to detect
const double THETA_SIM_THRESH = CV_PI / 30.0; //How similar two rows can be ----- 6 degrees
const int RHO_SIM_THRESH = 100;						 //How similar two rows can be
const double ANGLE_THRESH = CV_PI / 6.0;	//How steep angles the crop rows can be in radians -------- 30 degrees
const string FILE_NAME = "crop_row_013.JPG";

int main(int argc, char const *argv[])
{
	if(argc==2)
	{
		cout << "Read image from commandline\n";
		Mat im = imread(argv[1]);
		cropRowDetec(im);
	}
	if(argc == 1)
	{
		cout << "Read image from constant\n";
		Mat im = imread(FILE_NAME);
		cropRowDetec(im);
	}

	cout << "Done\n";
	return 0;
}

void cropRowDetec(Mat m)
{
	grayTransform(m);
	skeletonize(m);
	imshow("Skel", m);
	waitKey(0);
	houghTransform(m);
}

void grayTransform(Mat &m)
{

	for (size_t i = 0; i < m.rows; i++)
	{
		for (size_t j = 0; j < m.cols; j++)
		{
			
			Vec3b bgr = m.at<Vec3b>(i, j);
			auto GVal = 2 * bgr[1] - bgr[0] - bgr[1];
			m.at<Vec3b>(i, j)[0] = GVal;
			m.at<Vec3b>(i, j)[1] = GVal;
			m.at<Vec3b>(i, j)[2] = GVal;

		}
	}
	Mat temp;
	cv::cvtColor(m, temp, cv::COLOR_BGR2GRAY);
	m = temp;
}

void skeletonize(Mat &m)
{
	Mat binIm;
	cv::threshold(m, binIm, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	m = binIm;

	auto size = m.rows * m.cols;
	cv::Mat skel(m.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp(m.size(), CV_8UC1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done = false;
	while (!done)
	{
		Mat eroded;
		cv::erode(m, eroded, element);
		cv::dilate(eroded, temp, element);
		cv::subtract(m, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		m = eroded;
		auto zeros = size - cv::countNonZero(m);
		if (zeros == size)
		{
			done = true;
		}
	}

	m = skel;
}

void houghTransform(Mat &m)
{

	int hough_thresh = HOUGH_THRESH_MAX;
	bool row_found = false;

	vector<cv::Vec4i> lines;
	vector<cv::Vec4i> filteredLines;
	while(hough_thresh > HOUGH_THRESH_MIN && !row_found)
	{
		lines.clear();
		filteredLines.clear();
		cv::HoughLinesP(m, lines, HOUGH_RHO, HOUGH_ANGLE, hough_thresh, 0, 10);
		for (size_t i = 0; i < lines.size(); i++)
		{
			double xCor = lines[i][2] - lines[i][0];
			double yCor = lines[i][3] - lines[i][1];
			double angle = atan(abs(xCor / yCor));
			if (- ANGLE_THRESH < angle && angle < ANGLE_THRESH)
			{		
				filteredLines.push_back(cv::Vec4i(lines[i]));
			}
		}

		filterSimLines(filteredLines);

		hough_thresh -= HOUGH_THRESH_INCR;
		if(filteredLines.size() >= NUMBER_OF_ROWS)
		{
			row_found = true;
		}
	}

	Mat temp;
	cv::cvtColor(m, temp, cv::COLOR_GRAY2BGR);


	for (size_t i = 0; i < filteredLines.size(); i++)
	{
		cv::line(temp, cv::Point(filteredLines[i][0], filteredLines[i][1]), cv::Point(filteredLines[i][2], filteredLines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
	}
	imshow("Detected Lines", temp);
	waitKey(0);
}

double solveforB(const Vec4i &twoPts)
{
	if (twoPts[2] - twoPts[0] == 0)
	{
		return std::numeric_limits<double>::max();
	}
		double m = (twoPts[3] - twoPts[1]) / (twoPts[2] - twoPts[0]);
	return twoPts[3] - twoPts[2] * m;
}

void filterSimLines(vector<Vec4i> &lines)
{
	for (size_t i = lines.size() - 1; i > 0; i--)
	{
		double xCor = lines[i][2] - lines[i][0];
		double yCor = lines[i][3] - lines[i][1];
		double angle = atan(abs(xCor / yCor));
		double b1 = solveforB(lines[i]);
		for (size_t j = 0; j < lines.size(); j++)
		{
			if(j != i)
			{
				xCor = lines[j][2] - lines[j][0];
				yCor = lines[j][3] - lines[j][1];

				double angle2 = atan(abs(xCor / yCor));
				if (abs(angle - angle2) < THETA_SIM_THRESH)
				{
					double b2 = solveforB(lines[j]);
					if (abs(b1 - b2) < RHO_SIM_THRESH)
					{
						lines.erase(lines.begin() + i);
						break;
					}
				}
			}
			
		}
	}
}