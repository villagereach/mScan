#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "ImageProcessing.h"
#include <string>
#include <iostream>
#include <fstream>

#include "./ImageProcessing.h"

using namespace std;
using namespace cv;

#define DILATION 8
#define BLOCK_SIZE 3
#define DIST_PARAM 500

Mat comparison_vectors;
PCA my_PCA;
vector <bubble_val> training_bubble_values;
vector <Point2f> training_bubbles_locations;
float weight_param;

void configCornerArray(vector<Point2f>& corners, Point2f* corners_a);
void straightenImage(const Mat& input_image, Mat& output_image);
bubble_val checkBubble(Mat& det_img_gray, Point2f& bubble_location);

vector<bubble_val> ProcessImage(string &imagefilename, string &jsonfilename, float &weight) {
  weight_param = weight;
  vector < Point2f > corners, bubbles;
  vector<bubble_val> bubble_vals;
  vector<int> real_bubbles;
  Mat img, imgGrey, out, warped;
  string line;
  float segx, segy, bubx, buby;
  int bubble;

  // read the json file for bubble locations
  ifstream jsonfile(jsonfilename.c_str());
  if (jsonfile.is_open()) {
    while (getline(jsonfile, line)) {
      if (line.at(0) == '-') {
        line.erase(0, 1);
        stringstream ss(line);
        ss >> segx;
        ss >> segy;
      } else {
        stringstream ss(line);
        ss >> bubx;
        ss >> buby;
        Point2f bubble(segx+bubx, segy+buby);
        bubbles.push_back(bubble);
      }
    }
  }

  // read the file containing the actual bubble values
  ifstream bubblevalfile("bubble-val");
  if (bubblevalfile.is_open()) {
    while (getline(bubblevalfile, line)) {
      stringstream ss(line);
      while (ss >> bubble) {
        real_bubbles.push_back(bubble);
      }
    }
  }

  // Read the input image
  img = imread(imagefilename);
  if (img.data == NULL) {
    return vector<bubble_val>();
  }

  //cout << "converting to grayscale" << endl;
  cvtColor(img, imgGrey, CV_RGB2GRAY);

  Mat straightened_image(3300, 2550, CV_16U);

  //cout << "straightening image" << endl;
  straightenImage(imgGrey, straightened_image);
  
  //cout << "writing to output image" << endl;
  imwrite("straightened_" + imagefilename, straightened_image);
  
  //cout << "checking bubbles" << endl;
  vector<Point2f>::iterator it;
  for (it = bubbles.begin(); it != bubbles.end(); it++) {
    Scalar color(0, 0, 0);
    bubble_val current_bubble = checkBubble(straightened_image, *it);
    bubble_vals.push_back(current_bubble);
    rectangle(straightened_image, (*it)-Point2f(7,9), (*it)+Point2f(7,9), color);
  }

  imwrite("withbubbles_" + imagefilename, straightened_image);

  return bubble_vals;
}

void straightenImage(const Mat& input_image, Mat& output_image) {
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
  goodFeaturesToTrack(input_image_dilated, corners, 4, 0.01, DIST_PARAM, tmask, BLOCK_SIZE, false, 0.04);

  // Initialize the value of corners_a to that of orig_corners
  memcpy(corners_a, orig_corners, sizeof(orig_corners));
  configCornerArray(corners, corners_a);
  
  Mat H = getPerspectiveTransform(corners_a , orig_corners);
  warpPerspective(input_image, output_image, H, output_image.size());
}

/*
Takes a vector of corners and converts it into a properly formatted corner array.
Warning: destroys the corners vector in the process.
*/
void configCornerArray(vector<Point2f>& corners, Point2f* corners_a){
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
}

//Assuming there is a bubble at the specified location,
//this function will tell you if it is filled, empty, or probably not a bubble.
bubble_val checkBubble(Mat& det_img_gray, Point2f& bubble_location) {
  Mat query_pixels;
  getRectSubPix(det_img_gray, Point(14,18), bubble_location, query_pixels);
  query_pixels.convertTo(query_pixels, CV_32F);
  //normalize(query_pixels, query_pixels);
  transpose(my_PCA.project(query_pixels.reshape(0,1)), query_pixels);
  Mat M = (Mat_<float>(3,1) << weight_param, 1.0 - weight_param, 1.0);
  query_pixels = comparison_vectors*query_pixels.mul(M);
  Point max_location;
  minMaxLoc(query_pixels, NULL, NULL, NULL, &max_location);
  return training_bubble_values[max_location.y];
}

void train_PCA_classifier() {
  cout << "training PCA classifier" << endl;
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
}
