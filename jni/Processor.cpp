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

/**
 * Accepts an image file and the coordinates and size of a segment within the image.
 * This draws a rectangle representing the segment on the original image and creates
 * a new image file of the cropped segment.
 * filename - Absolute location of the original image file.
 * segment - Name of the segment being processed. This name will be used when
 *     creating output files for each segment.
 * x - x-coordinate of the segment's top left corner
 * y - y-coordinate of the segment's top left corner
 * width - the width of the segment
 * height - the height of the segment
 */
void Processor::processSegment(char* filename, char* segment, int x, int y, int width, int height) {
    string image_filename = string(filename);
    string segment_name = string(segment);

    Point2f top_left = Point2f(x, y); 
    Point2f bottom_right = Point2f(x + width, y + height);

    // Create the color used for highlighting segments.
    Scalar color = Scalar(0, 255, 0); 

    Mat in = imread(image_filename);

    // Draw a rectangle around the segment.
    rectangle(in, top_left, bottom_right, color, 1, 8, 0); 

    // Crop the segment
    Mat cropped_segment = in(Rect(x, y, width, height));

    // Write both images to file.
    imwrite(image_filename + "_highlighted.jpg", in);
    imwrite(image_filename + "_" + segment_name + ".jpg", cropped_segment);
}

enum bubble_val { EMPTY_BUBBLE, FILLED_BUBBLE, FALSE_POSITIVE };

#define SCALE .5f

Mat comparison_vectors;
PCA my_PCA;
vector <bubble_val> training_bubble_values;
vector <Point2f> training_bubbles_locations;
float weight_param;

void configCornerArray(vector<Point2f>& corners, Point2f* corners_a);
void straightenImage(const Mat& input_image, Mat& output_image);
bubble_val checkBubble(Mat& det_img_gray, Point2f& bubble_location);

/**
 * Function wrapper that calls the mScan ImageProcessing library.
 * imagefile - The absolute filepath of the image.
 * bubblefile - the absolute filepath of the bubble coordinates.
 * weight - unknown
 */
int Processor::processImage(char* imagefile, char* bubblesfile, float weight) {
	LOGI("Beginning to process image.");
  //string imagefilename(image);
  //string bubblefilename(bubblelist);

  string imagefilename(imagefile);
  string bubblefilename(bubblesfile);

  cout << "debug level is: " << DEBUG << endl;
  weight_param = weight;
  vector < Point2f > corners, bubbles;
  vector<bubble_val> bubble_vals;
  Mat img, imgGrey, out, warped;
  string line;
  float bubx, buby;
  int bubble;

  LOGI("READING BUBBLES");
  // read the bubble location file for bubble locations
  ifstream bubblefile(bubblefilename.c_str());
  if (bubblefile.is_open()) {
    while (getline(bubblefile, line)) {
      stringstream ss(line);
      ss >> bubx;
      ss >> buby;
      Point2f bubble(bubx * SCALE, buby * SCALE);
      bubbles.push_back(bubble);
    }
  }

  LOGI("done reading bubbles");

  // Read the input image
  img = imread(imagefilename);
  if (img.data == NULL) {
    //return vector<bubble_val>();
    return 0;
  }

  #if DEBUG > 0
  LOGI("BUBBLEBOT converting to grayscale");
  #endif
  cvtColor(img, imgGrey, CV_RGB2GRAY);

  Mat straightened_image(3300 * SCALE, 2550 * SCALE, CV_16U);

  #if DEBUG > 0
  LOGI("BUBBLEBOT straightening image");
  #endif
  straightenImage(imgGrey, straightened_image);
  
  #if DEBUG > 0
  LOGI("BUBBLEBOT writing to output image");
  imwrite("/mnt/sdcard/external_sd/straightened_" + imagefilename, straightened_image);
  #endif
  
  #if DEBUG > 0
  LOGI("BUBBLEBOT checking bubbles");
  #endif
  vector<Point2f>::iterator it;
  for (it = bubbles.begin(); it != bubbles.end(); it++) {
    Scalar color(0, 0, 0);
    LOGI("BUBBLEBOT in for loop. checking one bubble.");
    bubble_val current_bubble = checkBubble(straightened_image, *it);
    bubble_vals.push_back(current_bubble);
    rectangle(straightened_image, (*it)-Point2f(7,9), (*it)+Point2f(7,9), color);
  }

  #if DEBUG > 0
  imwrite("withbubbles_" + imagefilename, straightened_image);
  #endif

  //return bubble_vals;
  return 1;
}

void straightenImage(const Mat& input_image, Mat& output_image) {
  #if DEBUG > 0
  LOGI("BUBBLEBOT entering StraightenImage");
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
  LOGI("BUBBLEBOT dilating image");
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
  LOGI("BUBBLEBOT finding corners of the paper");
  #endif
  goodFeaturesToTrack(input_image_dilated, corners, 4, 0.01, DIST_PARAM, tmask, BLOCK_SIZE, false, 0.04);

  // Initialize the value of corners_a to that of orig_corners
  memcpy(corners_a, orig_corners, sizeof(orig_corners));
  configCornerArray(corners, corners_a);
  
  Mat H = getPerspectiveTransform(corners_a , orig_corners);
  #if DEBUG > 0
  LOGI("BUBBLEBOT resizing and warping");
  #endif
  warpPerspective(input_image, output_image, H, output_image.size());

  #if DEBUG > 0
  LOGI("BUBBLEBOT exiting StraightenImage");
  #endif
}

/*
Takes a vector of corners and converts it into a properly formatted corner array.
Warning: destroys the corners vector in the process.
*/
void configCornerArray(vector<Point2f>& corners, Point2f* corners_a){
  #if DEBUG > 0
  LOGI("BUBBLEBOT in configCornerArray");
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
  LOGI("BUBBLEBOT exiting configCornerArray");
  #endif
}

//Assuming there is a bubble at the specified location,
//this function will tell you if it is filled, empty, or probably not a bubble.
bubble_val checkBubble(Mat& det_img_gray, Point2f& bubble_location) {
  #if DEBUG > 1
  LOGI("BUBBLEBOT in checkBubble");
  cout << "checking bubble location: " << bubble_location << endl;
  #endif
  Mat query_pixels;
  getRectSubPix(det_img_gray, Point(14,18), bubble_location, query_pixels);
  query_pixels.convertTo(query_pixels, CV_32F);
  //normalize(query_pixels, query_pixels);
  transpose(my_PCA.project(query_pixels.reshape(0,1)), query_pixels);
  Mat M = (Mat_<float>(3,1) << weight_param, 1.0 - weight_param, 1.0);
  query_pixels = comparison_vectors*query_pixels.mul(M);
  Point max_location;
  minMaxLoc(query_pixels, NULL, NULL, NULL, &max_location);
  #if DEBUG > 1
  LOGI("BUBBLEBOT exiting checkBubble");
  #endif
  return training_bubble_values[max_location.y];
}

void train_PCA_classifier() {
  #if DEBUG > 0
  LOGI("BUBBLEBOT training PCA classifier");
  #endif
  //Goes though all the selected bubble locations and puts their pixels into rows of
  //a giant matrix called so we can perform PCA on them (we need at least 3 locations to do this)
  vector<Point2f> training_training_bubbles_locations;
  Mat train_img, train_img_gray;

  train_img = imread("training-image.jpg");
  if (train_img.data == NULL) {
    return;
  }

  cvtColor(train_img, train_img_gray, CV_RGB2GRAY);

  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubbles_locations.push_back(Point2f(35, 28));
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubbles_locations.push_back(Point2f(99, 35));
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubbles_locations.push_back(Point2f(113, 58));
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubbles_locations.push_back(Point2f(187, 11));
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubbles_locations.push_back(Point2f(200, 61));
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubbles_locations.push_back(Point2f(302, 58));
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubbles_locations.push_back(Point2f(276, 12));
  training_bubble_values.push_back(EMPTY_BUBBLE);
  training_bubbles_locations.push_back(Point2f(385, 36));
  training_bubble_values.push_back(FILLED_BUBBLE);
  training_bubbles_locations.push_back(Point2f(372, 107));
  Mat PCA_set = Mat::zeros(training_bubbles_locations.size(), 18*14, CV_32F);

  for(size_t i = 0; i < training_bubbles_locations.size(); i+=1) {
    Mat PCA_set_row;
    getRectSubPix(train_img_gray, Point(14,18), training_bubbles_locations[i], PCA_set_row);
    PCA_set_row.convertTo(PCA_set_row, CV_32F);
    normalize(PCA_set_row, PCA_set_row); //lighting invariance?
    PCA_set.row(i) += PCA_set_row.reshape(0,1);
  }

  my_PCA = PCA(PCA_set, Mat(), CV_PCA_DATA_AS_ROW, 3);
  comparison_vectors = my_PCA.project(PCA_set);
  for(size_t i = 0; i < comparison_vectors.rows; i+=1){
    comparison_vectors.row(i) /= norm(comparison_vectors.row(i));
  }

  #if DEBUG > 0
  LOGI("BUBBLEBOT finished training PCA classifier");
  #endif
}

