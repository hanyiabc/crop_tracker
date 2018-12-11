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
cv::Vec2d twoPoints2Polar(const cv::Vec4i &line);

const int HOUGH_RHO = 2;				 //Distance resolution of the accumulator in pixels
const double HOUGH_ANGLE = CV_PI / 45.0; //Angle resolution of the accumulator in radians --------- 4 degrees
const int HOUGH_THRESH_MAX = 100;			  //Accumulator threshold parameter.Only those lines are returned that get enough votes
const int HOUGH_THRESH_MIN = 10;
const int HOUGH_THRESH_INCR = 1;
const int NUMBER_OF_ROWS = 5;						 //how many crop rows to detect
const double THETA_SIM_THRESH = CV_PI / 30.0; //How similar two rows can be ----- 6 degrees
const int RHO_SIM_THRESH = 8;						 //How similar two rows can be
const double ANGLE_THRESH = CV_PI / 5.0;	//How steep angles the crop rows can be in radians -------- 30 degrees
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
		cv::HoughLinesP(m, lines, HOUGH_RHO, HOUGH_ANGLE, hough_thresh, 2, 50);
		
		filterSimLines(lines);

		hough_thresh -= HOUGH_THRESH_INCR;
		if(filteredLines.size() >= NUMBER_OF_ROWS)
		{
			row_found = true;
		}
	}

	Mat temp;
	cv::cvtColor(m, temp, cv::COLOR_GRAY2BGR);


	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::line(temp, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
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
	if (!lines.empty())
	{
		for (size_t i = lines.size() - 1; i > 0; i--)
		{
			auto rhoTheta = twoPoints2Polar(lines[i]);
			if ( ( (ANGLE_THRESH < rhoTheta[1]) && (rhoTheta[1] < CV_PI - ANGLE_THRESH) || (-ANGLE_THRESH > rhoTheta[1]) && (rhoTheta[1] > ANGLE_THRESH - CV_PI) ) || abs(rhoTheta[1]) <= 0.0001)
			{
				lines.erase(lines.begin() + i);
			}
			else
			{
				for (size_t j = 0; j < lines.size(); j++)
				{

					if (j != i)
					{
						auto rhoTheta2 = twoPoints2Polar(lines[j]);
						if (abs(rhoTheta[1] - rhoTheta2[1]) < THETA_SIM_THRESH)
						{
							lines.erase(lines.begin() + i);
							break;
						}
						else if (abs(rhoTheta[0] - rhoTheta2[0]) < RHO_SIM_THRESH)
						{
							lines.erase(lines.begin() + i);
							break;
						}
					}

				}

			}

		}
	}
	
}

cv::Vec2d twoPoints2Polar(const cv::Vec4i &line)
{
	// Get points from the vector
	cv::Point2f p1(line[0], line[1]);
	cv::Point2f p2(line[2], line[3]);

	// Compute 'rho' and 'theta'
	double rho = abs(p2.x * p1.y - p2.y * p1.x) / cv::norm(p2 - p1);
	double theta = -atan2((p2.x - p1.x), (p2.y - p1.y));

	// You can have a negative distance from the center
	// when the angle is negative
	if (theta < 0)
	{
		rho = -rho;
	}

	//if (theta < 0)
	//{
	//	theta = 2 * CV_PI + theta;
	//}
	return cv::Vec2d(rho, theta);
}