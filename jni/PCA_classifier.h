#ifndef PCA_CLASSIFIER_H
#define PCA_CLASSIFIER_H
#include "configuration.h"

#ifdef USE_ANDROID_HEADERS_AND_IO

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_pool.h"

#else
#include "cv.h"
#endif

//TODO: Figure out a way to set the example dimentions as variables.
//		They should scale relative to the size of the form speced in the template.
//		Additionally, they should probably be specified in the template.
//		(so perhaps scaling them to the template size is the template maker's problem?)
//TODO: I'm running into problems with the ex_width+height.
//		Should the width and height be measured from the edges of the bubble than have a additional buffer?
//		What if I want to specify arbitrary rectangles that should be resized then classified?
/*
#if 0
#define EXAMPLE_WIDTH 28
#define EXAMPLE_HEIGHT 36
#else
#define EXAMPLE_WIDTH 14
#define EXAMPLE_HEIGHT 18
#endif
*/
//TODO: Add a 4th classification. The middle classification doesn't really help since it has to be entirely
//		lumped with empty or full...
enum bubble_val { EMPTY_BUBBLE = 0, PARTIAL_BUBBLE, FILLED_BUBBLE, NUM_BUBBLE_VALS };

template <class Tp>
bool returnTrue(Tp& filename){
	return true;
}

class PCA_classifier
{
	cv::Mat comparison_vectors;
	std::vector <bubble_val> training_bubble_values;
	cv::PCA my_PCA;
	
	cv::Point search_window;

	//A matrix for precomputing gaussian weights for the search window
	cv::Mat gaussian_weights;

	//The weights Mat can be used to bias the classifier
	//Each element corresponds to a classification.
	cv::Mat weights;
	public:
		cv::Size exampleSize;
		PCA_classifier();
		void set_weight(bubble_val classification, float weight);
		void set_search_window(cv::Point sw);
		double rateBubble(cv::Mat& det_img_gray, cv::Point bubble_location);
		void train_PCA_classifier(cv::Size myExampleSize = cv::Size(14,18), bool (*pred)(std::string& filename) = &returnTrue);
		cv::Point bubble_align(cv::Mat& det_img_gray, cv::Point bubble_location);
		bubble_val classifyBubble(cv::Mat& det_img_gray, cv::Point bubble_location);
		
		virtual ~PCA_classifier() {
			//Not sure if I need to do anything here...
			//Do the class variables above automatically get destructed?
		}
	private:
		void update_gaussian_weights();
		void PCA_set_add(cv::Mat& PCA_set, cv::Mat& img);
		void PCA_set_add(cv::Mat& PCA_set, std::string& filename);
};

#endif