#include "Processor.h"
//#include "ImageProcessing.h"
#include <sys/stat.h>
#include "log.h"
#include <string>
#include <iostream>
#include <fstream>
#define LOG_COMPONENT "BubbleProcessor"

#define DILATION 6
#define BLOCK_SIZE 3
#define DIST_PARAM 500

#define DEBUG 5

using namespace std;
using namespace cv;

const char* const c_pszCaptureDir = "/sdcard/BubbleBot/capturedImages/";
const char* const c_pszProcessDir = "/sdcard/BubbleBot/processedImages/";
const char* const c_pszDataDir = "/sdcard/BubbleBot/processedText/";
const char* const c_pszTempJpgFilename = "/sdcard/BubbleBot/preview.jpg";
const char* const c_pszLastCannyFilename = "/sdcard/BubbleBot/lastCanny.jpg";
const char* const c_pszJpg = ".jpg";
const char* const c_pszTxt = ".txt";
const int c_leftMargin = 550;
const int c_rightMargin = 1500;

/* Processor
 *
 * This class detects the form in the received image and process the form image
 * to detects bubbles that are filled. It will then overlay the information on
 * the image as well as writing the digitized information to a text file.
 */

Processor::Processor() {
}

Processor::~Processor() {
}

// Reusing a function from feedback.cpp
extern float angle(Point pt1, Point pt2, Point pt0);

// DetectOutline
//
// This function detects the outline of a form in an image.
// Returns true if the outline is detected. False otherwise.
// The function writes the Canny-ed image to /sdcard/BubbleBot/lastCanny.jpg
// If the outline is detected, the function will:
// (1) Save the original image with the detected outline drawn in green on the image
//		to /sdcard/BubbleBot/preview.jpg
// (2) Create a text file <filename>.txt in the processedImages folder that
//		contains the data of the detected outline.
//
// filename - Filename of the input image
// fIgnoreDatFile - Set to true to avoid loading the outline data of an image if it has
//		already been processed by this function. By default, the function will look for
//		<filename>.txt in the processedImages folder. If the file exists, the function
//		will return the data from the file and skip the image processing to save time.
// outline - Out parameter that contains the detected rectangle
bool Processor::DetectOutline(char* filename, bool fIgnoreDatFile,
		Rect &outline) {
	bool fDetected = false;
	int maxContourArea = 100000;
	Mat img, imgGrey, imgCanny;
	Rect rectMax;
	vector < Point > approx;
	vector < vector<Point> > contours;
	vector < Vec4i > lines;

	// If the data file already exists, return its data and skip further processing.
	if (!fIgnoreDatFile) {
		string sInDatPath = c_pszProcessDir;
		sInDatPath += filename;
		sInDatPath += c_pszTxt;
		ifstream ifsDat(sInDatPath.c_str(), ifstream::in);
		if (ifsDat.good()) {
			if (ifsDat >> outline.x >> outline.y >> outline.width
					>> outline.height) {
				ifsDat.close();
				return true;
			}
		}
	}

	// Read the input image
	string sInFilePath = c_pszCaptureDir;
	sInFilePath += filename;
	sInFilePath += c_pszJpg;
	img = imread(sInFilePath);
	if (img.data == NULL) {
		char msg[100];
		sprintf(msg, "DetectOutline: Failed to read file %s",
				sInFilePath.c_str());
		LOGE(msg);
		return 0;
	}

	// Convert the image to greyscale
	cvtColor(img, imgGrey, CV_RGB2GRAY);

	// Perform Canny transformation on the image
	Canny(imgGrey, imgCanny, 80, 80 * 3.5, 3);
	imwrite(c_pszLastCannyFilename, imgCanny);

	// Emphasize lines in the transformed image
	HoughLinesP(imgCanny, lines, 1, CV_PI / 180, 80, 700, 200);
	for (size_t i = 0; i < lines.size(); i++) {
		line(imgCanny, Point(lines[i][0], lines[i][1]), Point(lines[i][2],
				lines[i][3]), Scalar(255, 255, 255), 1, 8);
	}

	// Find all external contours of the image
	findContours(imgCanny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Iterate through all detected contours and find the biggest rectangle
	for (size_t i = 0; i < contours.size(); ++i) {
		Rect rectCur = boundingRect(Mat(contours[i]));
		int area = (rectCur.height * rectCur.width);
		if (area > maxContourArea) {
			rectMax = rectCur;
			maxContourArea = area;
			fDetected = true;
		}
	}

	// If outline is detected, draw it on the original image and
	// save that to the preview jpg file. Also, write the outline
	// rectangle information to a data file.
	if (fDetected) {
		LOGI("DetectOutline: Outline detected");
		rectangle(img, Point(rectMax.x, rectMax.y), Point(rectMax.x
				+ rectMax.width, rectMax.y + rectMax.height),
				Scalar(0, 255, 0), 4);
		imwrite(c_pszTempJpgFilename, img);

		string sOutDataFilePath = c_pszProcessDir;
		sOutDataFilePath += filename;
		sOutDataFilePath += c_pszTxt;
		ofstream ofsData(sOutDataFilePath.c_str(), ios_base::trunc);
		ofsData << rectMax.x << " " << rectMax.y << " " << rectMax.width << " "
				<< rectMax.height;
		ofsData.close();
		return true;
	}
	LOGE("DetectOutline: Failed to detect outline");
	return false;
}

// DetectOutline
//
// This is another prototype of the function that does not require a Rect
// as an input parameter. This function is intended to be called from Java.
bool Processor::DetectOutline(char* filename, bool fIgnoreDatFile) {
	Rect r;
	return DetectOutline(filename, fIgnoreDatFile, r);
}

// Digitize the given bubble form
char* Processor::ProcessForm(char* filename) {
	string fullname = c_pszCaptureDir;
	fullname += filename;
	fullname += c_pszJpg;

	LOGI("Entering ProcessForm()");

	//Load image
	IplImage *img = cvLoadImage(fullname.c_str());
	if (!img) {
		LOGE("Image load failed");
		return NULL;
	}

	Rect rectBorder;
	DetectOutline(filename, false, rectBorder);

	CvPoint * cornerPoints = new CvPoint[4];
	cornerPoints[0].x = rectBorder.x;
	cornerPoints[0].y = rectBorder.y;
	cornerPoints[1].x = rectBorder.x + rectBorder.width;
	cornerPoints[1].y = rectBorder.y;
	cornerPoints[2].x = rectBorder.x + rectBorder.width;
	cornerPoints[2].y = rectBorder.y + rectBorder.height;
	cornerPoints[3].x = rectBorder.x;
	cornerPoints[3].y = rectBorder.y + rectBorder.height;

	IplImage* warpImg = cvCreateImage(cvSize(img->width, img->height),
			img->depth, img->nChannels);
	warpImage(img, warpImg, cornerPoints);

	//Detect form squares
	CvPoint* lineValues = new CvPoint[5];
	lineValues = findLineValues(warpImg);

	//Find bubbles
	vector < Point > bubbles = findBubbles(warpImg);

	//Count bubbles
	int * count = new int[5];
	count[0] = 0;
	count[1] = 0;
	count[2] = 0;
	count[3] = 0;
	count[4] = 0;

	cvLine(warpImg, cvPoint(c_leftMargin, 0), cvPoint(c_leftMargin, warpImg->height), cvScalar(0,
			0, 255), 3, CV_AA, 0);
	cvLine(warpImg, cvPoint(c_rightMargin, 0), cvPoint(c_rightMargin, warpImg->height), cvScalar(
			0, 0, 255), 3, CV_AA, 0);

	// Draw detected bubbles in the image
	int i;
	for (i = 0; i < bubbles.size(); i++) {
		for (int j = 0; j < 5; ++j) {
			if ((bubbles[i].y > lineValues[j].y) && (bubbles[i].y
					< lineValues[j].y + 200)) {
				++count[j];
				bubbles[i].x += c_leftMargin;
				cvCircle(warpImg, bubbles[i], 20, cvScalar(0, 0, 255), 3,
						CV_AA, 0);
				break;
			}
		}
	}

	//Draw Text
	CvFont font;
	double hScale = 3.0;
	double vScale = 3.0;
	int lineWidth = 5;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale,
			0, lineWidth);

	for (i = 0; i < 5; i++) {
		int Y = lineValues[i].y + 130;
		char * number = new char[5];
		sprintf(number, "=%i", count[i]);
		cvPutText(warpImg, number, cvPoint(c_rightMargin, Y), &font, cvScalar(255, 0, 0));
	}

	//Draw results
	cvLine(img, cornerPoints[0], cornerPoints[1], cvScalar(0, 255, 0), 5);
	cvLine(img, cornerPoints[1], cornerPoints[2], cvScalar(0, 255, 0), 5);
	cvLine(img, cornerPoints[2], cornerPoints[3], cvScalar(0, 255, 0), 5);
	cvLine(img, cornerPoints[3], cornerPoints[0], cvScalar(0, 255, 0), 5);

	cvCircle(img, cornerPoints[0], 20, cvScalar(0, 255, 0), 5);
	cvCircle(img, cornerPoints[1], 20, cvScalar(0, 255, 0), 5);
	cvCircle(img, cornerPoints[2], 20, cvScalar(0, 255, 0), 5);
	cvCircle(img, cornerPoints[3], 20, cvScalar(0, 255, 0), 5);

	//Save image
	string saveLocation = c_pszProcessDir;
	saveLocation += filename;
	saveLocation += c_pszJpg;
	cvSaveImage(saveLocation.c_str(), warpImg);

	/*open a file in text mode with write permissions.*/
	string fileLocation = c_pszDataDir;
	fileLocation += filename;
	fileLocation += c_pszTxt;

	FILE *file = fopen(fileLocation.c_str(), "wt");
	if (file == NULL) {
		//If unable to open the specified file display error and return.
		LOGE("Failed to save text file");
		return NULL;
	}

	//Print some random text for now
	fprintf(file, "Vaccination Report\n");
	fprintf(file, "Date: 03/07/2011\n");
	fprintf(file, "Total BCG: %i\n", count[0]);
	fprintf(file, "Total Polio: %i\n", count[1]);
	fprintf(file, "Total Measles: %i\n", count[2]);
	fprintf(file, "Total Hepatitis B: %i\n", count[3]);
	fprintf(file, "Total Hepatitis B: %i\n", count[4]);
	fprintf(file, "Total All: %i\n", count[0] + count[1] + count[2] + count[3]
			+ count[4]);

	//release the file pointer.
	fclose(file);

	cvReleaseImage(&img);
	cvReleaseImage(&warpImg);

	LOGI("Exiting ProcessForm()");
	return filename;
}

// This function identifies bubbles in an image and checks whether
// those bubbles are "filled" or not.
vector<Point> Processor::findBubbles(IplImage* pImage) {
	Mat img(pImage), imgCropped, imgGrey;
	vector < Vec3f > circles;
	vector < Point > result;

	// Crop the image to focus on the area where the bubbles are.
	// This significantly speeds up processing time since there are
	// usually a lot of noise on other parts of the image.
	Size cropSize(c_rightMargin - c_leftMargin, img.rows);
	Point cropCenter(c_leftMargin + (c_rightMargin - c_leftMargin) / 2,
			img.rows / 2);
	getRectSubPix(img, cropSize, cropCenter, imgCropped);

	// Convert the image to greyscale
	cvtColor(imgCropped, imgGrey, CV_RGB2GRAY);

	// Blur the image to reduce noise in circle detection. A 3x3 size works better
	// but 5x5 is used to speed up the process
	GaussianBlur(imgGrey, imgGrey, Size(3, 3), 2, 2);

	// Detects circles using the Hough Circles algorithm
	//HoughCircles(imgGrey, circles, CV_HOUGH_GRADIENT, 2, 75, 20, 15, 6, 16);
	//HoughCircles(imgGrey, circles, CV_HOUGH_GRADIENT, 2, 75, 20, 5, 5, 20);
	HoughCircles(imgGrey, circles, CV_HOUGH_GRADIENT, 2, 50, 10, 5, 5, 20);

	// Process each detected circle
	for (size_t i = 0; i < circles.size(); i++) {
		Point center(cvRound( circles[i][0]), cvRound( circles[i][1]));

		// Extract the circle to a new image
		int radius = cvRound(circles[i][2]);
		Size patchSize(radius * 2, radius * 2);
		Mat imgCircle(patchSize, imgGrey.type());
		getRectSubPix(imgGrey, patchSize, center, imgCircle);

		// Apply histogram on that image to tally the values in 2 bins
		MatND hist;
		int channels[] = { 0 };
		int histSize[] = { 2 };
		float range[] = { 0, 150 };
		const float *ranges[] = { range };
		calcHist(&imgCircle, 1, channels, Mat(), hist, 1, histSize, ranges,
				true, false);

		// If the dark bin has more count than the light bin, the bubble
		// is filled. Add the bubble information to the result list.
		if (hist.at<float> (0) > hist.at<float> (1)) {
			result.push_back(center);
		}
	}

	return result;
}

// This function searches for the top line in an image to determine the
// vertical positions of the 5 areas in the form.
CvPoint * Processor::findLineValues(IplImage* img) {
	CvPoint* lineValues = new CvPoint[5];
	IplImage *warpImg = cvCloneImage(img);
	CvRect rect;
	rect = cvRect(1500, 0, 100, 400);
	cvSetImageROI(warpImg, rect);

	//params for Canny
	int N = 7;
	double lowThresh = 50;
	double highThresh = 300;

	// Apply Canny filter on the image
	IplImage* bChannel = cvCreateImage(cvGetSize(warpImg), warpImg->depth, 1);
	cvCvtPixToPlane(warpImg, bChannel, NULL, NULL, NULL);
	IplImage* out = cvCreateImage(cvGetSize(bChannel), bChannel->depth,
			bChannel->nChannels);
	cvCanny(bChannel, out, lowThresh * N * N, highThresh * N * N, N);

	// Find edge
	int maxWhiteCount = 0;
	int linePoint = 0;
	int j, k;
	int whiteCount;
	CvScalar s;
	for (j = 0; j < out->height; j++) {
		whiteCount = 0;
		for (k = 0; k < out->width; k++) {
			s = cvGet2D(out, j, k);
			if (s.val[0] == 255) {
				whiteCount++;
			}
		}
		if (whiteCount > maxWhiteCount) {
			maxWhiteCount = whiteCount;
			linePoint = j;
		}
	}

	if (linePoint < 100)
	{
		linePoint = 250;
	}
	// Calibrate the top line position and use that to
	// determine the position for the other 5 lines
	linePoint = linePoint - 60;
	lineValues[0].x = 0;
	lineValues[0].y = linePoint;
	lineValues[1].x = 0;
	lineValues[1].y = linePoint + 260;
	lineValues[2].x = 0;
	lineValues[2].y = linePoint + 510;
	lineValues[3].x = 0;
	lineValues[3].y = linePoint + 770;
	lineValues[4].x = 0;
	lineValues[4].y = linePoint + 1040;

	cvReleaseImage(&warpImg);
	cvReleaseImage(&bChannel);
	cvReleaseImage(&out);
	return lineValues;
}

// This function crop the the image to the form area
void Processor::warpImage(IplImage* img, IplImage* warpImg,
		CvPoint * cornerPoints) {
	CvPoint2D32f templatePoint[4], currentPoint[4];

	templatePoint[0].x = 0;
	templatePoint[0].y = 0;
	templatePoint[1].x = img->width;
	templatePoint[1].y = 0;
	templatePoint[2].x = img->width;
	templatePoint[2].y = img->height;
	templatePoint[3].x = 0;
	templatePoint[3].y = img->height;

	currentPoint[0].x = cornerPoints[0].x;
	currentPoint[0].y = cornerPoints[0].y;
	currentPoint[1].x = cornerPoints[1].x;
	currentPoint[1].y = cornerPoints[1].y;
	currentPoint[2].x = cornerPoints[2].x;
	currentPoint[2].y = cornerPoints[2].y;
	currentPoint[3].x = cornerPoints[3].x;
	currentPoint[3].y = cornerPoints[3].y;

	CvMat* map = cvCreateMat(3, 3, CV_32FC1);
	cvGetPerspectiveTransform(templatePoint, currentPoint, map);
	cvWarpPerspective(img, warpImg, map, CV_WARP_FILL_OUTLIERS
			+ CV_WARP_INVERSE_MAP, cvScalar(0, 0, 0));

	//Reduce search space - cut white off form edge
	CvRect rect;
	rect = cvRect(100, 50, warpImg->width - 200, warpImg->height - 75);
	cvSetImageROI(warpImg, rect);
	cvReleaseMat(&map);
}

/*
 * processing constants
 */
#define DILATION 6
#define BLOCK_SIZE 3
#define DIST_PARAM 500

#define EXAMPLE_WIDTH 26
#define EXAMPLE_HEIGHT 32

// how wide is the segment in pixels
#define SEGMENT_WIDTH 144

// how tall is the segment in pixels
#define SEGMENT_HEIGHT 200

// buffer around segment in pixels
#define SEGMENT_BUFFER 70

#define EIGENBUBBLES 5

enum bubble_val { EMPTY_BUBBLE, FILLED_BUBBLE, FALSE_POSITIVE };

Mat comparison_vectors;
PCA my_PCA;
vector <bubble_val> training_bubble_values;
vector <Point2f> training_bubbles_locations;
float weight_param;
string imgfilename;
Point search_window(10, 10);

void configCornerArray(vector<Point2f>& corners, Point2f* corners_a);
void straightenImage(const Mat& input_image, Mat& output_image);
double rateBubble(Mat& det_img_gray, Point bubble_location);
bubble_val checkBubble(Mat& det_img_gray, Point bubble_location);
void getSegmentLocations(vector<Point2f> &segmentcorners, string segfile);
vector<bubble_val> processSegment(Mat &segment, string bubble_offsets);
Mat getSegmentMat(Mat &img, Point2f &corner);
void find_bounding_lines(Mat& img, int* upper, int* lower, bool vertical);
void align_segment(Mat& img, Mat& aligned_segment);

template <class Tp>
void configCornerArray(vector<Tp>& orig_corners, Point2f* corners_a, float expand) {
  float min_dist;
  int min_idx;
  float dist;
  
  vector<Point2f> corners;
  
  for(int i = 0; i < orig_corners.size(); i++ ){
    corners.push_back(Point2f(float(orig_corners[i].x), float(orig_corners[i].y)));
  }
  //Make sure the form corners map to the correct image corner
  //by snaping the nearest form corner to each image corner.
  for(int i = 0; i < 4; i++) {
    min_dist = FLT_MAX;
    for(int j = 0; j < corners.size(); j++ ){
      dist = norm(corners[j]-corners_a[i]);
      if(dist < min_dist){
        min_dist = dist;
        min_idx = j;
      }
    }
    corners_a[i]=corners[min_idx] + expand * (corners_a[i] - corners[min_idx]);
    //Two relatively minor reasons for doing this,
    //1. Small efficiency gain
    //2. If 2 corners share a closest point this resolves the conflict.
    corners.erase(corners.begin()+min_idx);
  }
}

int Processor::processImage(char* image_file_name, char* bubble_file_name,
    float weight) {

  string imagefilename(image_file_name);
  string bubblefilename(bubble_file_name);

  #if DEBUG > 0
  cout << "debug level is: " << DEBUG << endl;
  #endif
  string seglocfile("segment-offsets-tmp.txt");
  string buboffsetfile("bubble-offsets.txt");
  weight_param = weight;
  vector < Point2f > corners, segment_locations;
  vector<bubble_val> bubble_vals;
  vector<Mat> segmats;
  vector<vector<bubble_val> > segment_results;
  Mat img, imgGrey, out, warped;
  imgfilename = imagefilename;

  // Read the input image
  img = imread(imagefilename);
  if (img.data == NULL) {
    // return vector<vector<bubble_val> >();
    return 0;
  }

  #if DEBUG > 0
  cout << "converting to grayscale" << endl;
  #endif
  cvtColor(img, imgGrey, CV_RGB2GRAY);

  Mat straightened_image(3300, 2550, CV_16U);

  #if DEBUG > 0
  cout << "straightening image" << endl;
  #endif
  straightenImage(imgGrey, straightened_image);
  
  #if DEBUG > 0
  cout << "writing to output image" << endl;
  imwrite("straightened_" + imagefilename, straightened_image);
  #endif
  
  #if DEBUG > 0
  cout << "getting segment locations" << endl;
  #endif
  getSegmentLocations(segment_locations, seglocfile);

  #if DEBUG > 0
  cout << "grabbing individual segment images" << endl;
  #endif
  for (vector<Point2f>::iterator it = segment_locations.begin();
       it != segment_locations.end(); it++) {
    segmats.push_back(getSegmentMat(straightened_image, *it));
  }

  #if DEBUG > 0
  cout << "processing all segments" << endl;
  #endif
  for (vector<Mat>::iterator it = segmats.begin(); it != segmats.end(); it++) {
    //wouldn't be a bad idea memory wise to merge this loop with the above loop
    Mat aligned_segment((*it).rows - SEGMENT_BUFFER, (*it).cols - SEGMENT_BUFFER, CV_8UC1);

    align_segment(*it, aligned_segment);
    segment_results.push_back(processSegment(aligned_segment, buboffsetfile));
    aligned_segment.copyTo(*it);
  }

  #if DEBUG > 0
  cout << "writing segment images" << endl;
  for (int i = 0; i < segmats.size(); i++) {
    #if DEBUG > 1
    cout << "writing segment " << i << endl;
    #endif
    string segfilename("marked_");
    segfilename.push_back((char)i+33);
    segfilename.append(".jpg");
    imwrite(segfilename, segmats[i]);
  }
  #endif

  //return segment_results;
  return 1;
}

void align_segment(Mat& img, Mat& aligned_segment){
  vector < vector<Point> > contours;
  vector < vector<Point> > borderContours;
  vector < Point > approx;
  vector < Point > maxRect;

  //Threshold the image
  //Maybe we should dilate or blur or something first?
  //The ideal image would be black lines and white boxes with nothing in them
  //so if we can filter to get something closer to that, it is a good thing.
  Scalar my_mean;
  Scalar my_stddev;
  meanStdDev(img, my_mean, my_stddev);
  Mat imgThresh = img > (my_mean.val[0]-.05*my_stddev.val[0]);

  // Find all external contours of the image
  findContours(imgThresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

  float maxContourArea = 0;
  // Iterate through all detected contours
  for (size_t i = 0; i < contours.size(); ++i) {
    // reduce the number of points in the contour
    approxPolyDP(Mat(contours[i]), approx,
                     arcLength(Mat(contours[i]), true) * 0.1, true);

    float area = fabs(contourArea(Mat(approx)));

    if (area > maxContourArea) {
      maxRect = approx;
            maxContourArea = area;
        }
        //Maybe I could refine this by using a corner detector and using
        //the 4 contour points with the highest responce?
    }
    if ( maxRect.size() == 4 && isContourConvex(Mat(maxRect)) && maxContourArea > (img.cols/2) * (img.rows/2)) {
        Point2f segment_corners[4] = {Point2f(0,0),Point2f(aligned_segment.cols,0),
            Point2f(0,aligned_segment.rows),Point2f(aligned_segment.cols,aligned_segment.rows)};
        Point2f corners_a[4] = {Point2f(0,0),Point2f(img.cols,0),
            Point2f(0,img.rows),Point2f(img.cols,img.rows)};

        configCornerArray(maxRect, corners_a, .1);
        Mat H = getPerspectiveTransform(corners_a , segment_corners);
        warpPerspective(img, aligned_segment, H, aligned_segment.size());
    }
    else{//use the bounding line method if the contour method fails
        int top = 0, bottom = 0, left = 0, right = 0;
        find_bounding_lines(img, &top, &bottom, false);
        find_bounding_lines(img, &left, &right, true);
        
        //debug stuff
        /*
        img.copyTo(aligned_segment);
        const Point* p = &maxRect[0];
        int n = (int) maxRect.size();
        polylines(aligned_segment, &p, &n, 1, true, 200, 2, CV_AA);
        
        img.row(top)+=200;
        img.row(bottom)+=200;
        img.col(left)+=200;
        img.col(right)+=200;
        img.copyTo(aligned_segment);
        */

        float bounding_lines_threshold = .2;
        if ((abs((bottom - top) - aligned_segment.rows) < bounding_lines_threshold * aligned_segment.rows) &&
            (abs((right - left) - aligned_segment.cols) < bounding_lines_threshold * aligned_segment.cols) &&
            top + aligned_segment.rows < img.rows &&  top + aligned_segment.cols < img.cols) {
            
            img(Rect(left, top, aligned_segment.cols, aligned_segment.rows)).copyTo(aligned_segment);
            
        }
        else{
            img(Rect(SEGMENT_BUFFER, SEGMENT_BUFFER, aligned_segment.cols, aligned_segment.rows)).copyTo(aligned_segment);
        }
    }
}

vector<bubble_val> processSegment(Mat &segment, string bubble_offsets) {
  vector<Point2f> bubble_locations;
  vector<bubble_val> retvals;
  string line;
  float bubx, buby;
  ifstream offsets(bubble_offsets.c_str());

  if (offsets.is_open()) {
    while (getline(offsets, line)) {
      if (line.size() > 2) {
        stringstream ss(line);

        ss >> bubx;
        ss >> buby;
        Point2f bubble(bubx, buby + 3);
        bubble_locations.push_back(bubble);
      }
    }
  }

  vector<Point2f>::iterator it;
  for (it = bubble_locations.begin(); it != bubble_locations.end(); it++) {
    bubble_val current_bubble = checkBubble(segment, *it);
    Scalar color(0, 0, 0);
    if (current_bubble == 1) {
      color = (255, 255, 255);
    }
    rectangle(segment, (*it)-Point2f(EXAMPLE_WIDTH/2,EXAMPLE_HEIGHT/2),
              (*it)+Point2f(EXAMPLE_WIDTH/2,EXAMPLE_HEIGHT/2), color);

    retvals.push_back(current_bubble);
  }

  return retvals;
}

Mat getSegmentMat(Mat &img, Point2f &corner) {
  Mat segment;
  Point2f segcenter;
  segcenter += corner;
  segcenter.x += SEGMENT_WIDTH/2;
  segcenter.y += SEGMENT_HEIGHT/2;
  Size segsize(SEGMENT_WIDTH + SEGMENT_BUFFER, SEGMENT_HEIGHT + SEGMENT_BUFFER);
  getRectSubPix(img, segsize, segcenter, segment);

  #if DEBUG > 1
  string pcorner;
  pcorner.push_back(corner.x);
  pcorner.append("-");
  pcorner.push_back(corner.y);
  imwrite("segment_" + pcorner + "_" + imgfilename, segment);
  #endif

  return segment;
}

void getSegmentLocations(vector<Point2f> &segmentcorners, string segfile) {
  string line;
  float segx = 0, segy = 0;

  ifstream segstream(segfile.c_str());
  if (segstream.is_open()) {
    while (getline(segstream, line)) {
      if (line.size() > 2) {
        stringstream ss(line);

        ss >> segx;
        ss >> segy;
        Point2f corner(segx, segy);
        #if DEBUG > 1
        cout << "adding segment corner " << corner << endl;
        #endif
        segmentcorners.push_back(corner);
      }
    }
  }
}

void find_bounding_lines(Mat& img, int* upper, int* lower, bool vertical) {
  int center_size = 20;
  Mat grad_img, out;
  Sobel(img, grad_img, 0, int(!vertical), int(vertical));
  //multiply(grad_img, img/100, grad_img);//seems to yield improvements on bright images
  reduce(grad_img, out, int(!vertical), CV_REDUCE_SUM, CV_32F);
  GaussianBlur(out, out, Size(1,3), 1.0);

  if( vertical )
    transpose(out,out);

  Point min_location_top;
  Point min_location_bottom;
  minMaxLoc(out(Range(3, out.rows/2 - center_size), Range(0,1)), NULL,NULL,&min_location_top);
  minMaxLoc(out(Range(out.rows/2 + center_size,out.rows - 3), Range(0,1)), NULL,NULL,&min_location_bottom);
  *upper = min_location_top.y;
  *lower = min_location_bottom.y + out.rows/2 + center_size;
}

void straightenImage(const Mat& input_image, Mat& output_image) {
  #if DEBUG > 0
  cout << "entering StraightenImage" << endl;
  #endif
  Point2f orig_corners[4];
  Point2f corners_a[4];
  vector < Point2f > corners;

  Mat tmask, input_image_dilated;

  // Create a mask that limits corner detection to the corners of the image.
  tmask= Mat::zeros(input_image.rows, input_image.cols, CV_8U);
  circle(tmask, Point(0,0), (tmask.cols+tmask.rows)/8, Scalar(255,255,255), -1);
  circle(tmask, Point(0,tmask.rows), (tmask.cols+tmask.rows)/8, Scalar(255,255,255), -1);
  circle(tmask, Point(tmask.cols,0), (tmask.cols+tmask.rows)/8, Scalar(255,255,255), -1);
  circle(tmask, Point(tmask.cols,tmask.rows), (tmask.cols+tmask.rows)/8, Scalar(255,255,255), -1);

  //orig_corners = {Point(0,0),Point(img.cols,0),Point(0,img.rows),Point(img.cols,img.rows)};
  orig_corners[0] = Point(0,0);
  orig_corners[1] = Point(output_image.cols,0);
  orig_corners[2] = Point(0,output_image.rows);
  orig_corners[3] = Point(output_image.cols,output_image.rows);

  #if DEBUG > 0
  cout << "dilating image" << endl;
  #endif
  // Dilating reduces noise, thin lines and small marks.
  dilate(input_image, input_image_dilated, Mat(), Point(-1, -1), DILATION);

  /*
  Params for goodFeaturesToTrack:
  Source Mat, Dest Mat
  Number of features/interest points to return
  Minimum feature quality
  Min distance between corners (Probably needs parameterization depending on im. res. and form)
  Mask
  Block Size (not totally sure but I think it's similar to aperture)
  Use Harris detector (true) or cornerMinEigenVal() (false)
  Free parameter of Harris detector
  */
  #if DEBUG > 0
  cout << "finding corners of the paper" << endl;
  #endif
  goodFeaturesToTrack(input_image_dilated, corners, 4, 0.01, DIST_PARAM, tmask, BLOCK_SIZE, false, 0.04);

  // Initialize the value of corners_a to that of orig_corners
  memcpy(corners_a, orig_corners, sizeof(orig_corners));
  configCornerArray(corners, corners_a);
  
  Mat H = getPerspectiveTransform(corners_a , orig_corners);
  #if DEBUG > 0
  cout << "resizing and warping" << endl;
  #endif
  warpPerspective(input_image, output_image, H, output_image.size());

  #if DEBUG > 0
  cout << "exiting StraightenImage" << endl;
  #endif
}

/*
Takes a vector of corners and converts it into a properly formatted corner array.
Warning: destroys the corners vector in the process.
*/
void configCornerArray(vector<Point2f>& corners, Point2f* corners_a){
  #if DEBUG > 0
  cout << "in configCornerArray" << endl;
  #endif
  float min_dist;
  int min_idx;
  float dist;
  
  //Make sure the form corners map to the correct image corner
  //by snaping the nearest form corner to each image corner.
  for(int i = 0; i < 4; i++) {
    min_dist = FLT_MAX;
    for(int j = 0; j < corners.size(); j++ ){
      dist = norm(corners[j]-corners_a[i]);
      if(dist < min_dist){
        min_dist = dist;
        min_idx = j;
      }
    }
    corners_a[i]=corners[min_idx];
    //Two relatively minor reasons for doing this,
    //1. Small efficiency gain
    //2. If 2 corners share a closest point this resolves the conflict.
    corners.erase(corners.begin()+min_idx);
  }

  #if DEBUG > 0
  cout << "exiting configCornerArray" << endl;
  #endif
}

//Rate a location on how likely it is to be a bubble
double rateBubble(Mat& det_img_gray, Point bubble_location) {
    Mat query_pixels, pca_components;
    getRectSubPix(det_img_gray, Point(EXAMPLE_WIDTH,EXAMPLE_HEIGHT), bubble_location, query_pixels);
    query_pixels.reshape(0,1).convertTo(query_pixels, CV_32F);
    pca_components = my_PCA.project(query_pixels);
    //The rating is the SSD of query pixels and their back projection
    Mat out = my_PCA.backProject(pca_components)- query_pixels;
    return sum(out.mul(out)).val[0];
}

//Compare the bubbles with all the bubbles used in the classifier.
bubble_val checkBubble(Mat& det_img_gray, Point bubble_location) {
    Mat query_pixels;
    //This bit of code finds the location in the search_window most likely to be a bubble
    //then it checks that rather than the exact specified location.
    Mat out = Mat::zeros(Size(search_window.y, search_window.x) , CV_32FC1);
    Point offset = Point(bubble_location.x - search_window.x/2, bubble_location.y - search_window.y/2);
    for(size_t i = 0; i < search_window.y; i+=1) {
        for(size_t j = 0; j < search_window.x; j+=1) {
            out.row(i).col(j) += rateBubble(det_img_gray, Point(j,i) + offset);
        }
    }
    Point min_location;
    minMaxLoc(out, NULL,NULL, &min_location);

    getRectSubPix(det_img_gray, Point(EXAMPLE_WIDTH,EXAMPLE_HEIGHT), min_location + offset, query_pixels);

    query_pixels.reshape(0,1).convertTo(query_pixels, CV_32F);
   
    Mat responce;
    matchTemplate(comparison_vectors, my_PCA.project(query_pixels), responce, CV_TM_CCOEFF_NORMED);
   
    //Here we find the best match in our PCA training set with weighting applied.
    reduce(responce, out, 1, CV_REDUCE_MAX);
    int max_idx = -1;
    float max_responce = 0;
    for(size_t i = 0; i < training_bubble_values.size(); i+=1) {
        float current_responce = sum(out.row(i)).val[0];
        switch( training_bubble_values[i] ){
            case FILLED_BUBBLE:
                current_responce *= weight_param;
                break;
            case EMPTY_BUBBLE:
                current_responce *= (1 - weight_param);
                break;
        }
        if(current_responce > max_responce){
            max_idx = i;
            max_responce = current_responce;
        }
    }

    return training_bubble_values[max_idx];
}

void train_PCA_classifier() {

 // Set training_bubble_values here
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubble_values.push_back(FILLED_BUBBLE);

  Mat example_strip = imread("example_strip.jpg");
  Mat example_strip_bw;
  cvtColor(example_strip, example_strip_bw, CV_RGB2GRAY);

  int numexamples = example_strip_bw.cols / EXAMPLE_WIDTH;
  Mat PCA_set = Mat::zeros(numexamples, EXAMPLE_HEIGHT*EXAMPLE_WIDTH, CV_32F);

  for (int i = 0; i < numexamples; i++) {
    Mat PCA_set_row = example_strip_bw(Rect(i * EXAMPLE_WIDTH, 0,
                                            EXAMPLE_WIDTH, EXAMPLE_HEIGHT));
    PCA_set_row.convertTo(PCA_set_row, CV_32F);
    PCA_set.row(i) += PCA_set_row.reshape(0,1);
  }

  my_PCA = PCA(PCA_set, Mat(), CV_PCA_DATA_AS_ROW, EIGENBUBBLES);
  comparison_vectors = my_PCA.project(PCA_set);
}
