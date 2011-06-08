/*
 * Header file for image processing functions.
 */
#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>
#include <string>

class ImageProcessor {
    public:
        ImageProcessor();
        virtual ~ImageProcessor();

        enum bubble_val { EMPTY_BUBBLE, FILLED_BUBBLE, FALSE_POSITIVE };

        // takes a filename and processes the entire image for bubbles
        //int ProcessImage(char* imagefilename, char* bubblefilename, float weight);
        int mscan();

        // trains the program what bubbles look like
        void train_PCA_classifier();

        // takes a filename and JSON spec and looks for bubbles according
        // to locations coded in the JSON
        // int ProcessImage(string image, JSON_OBJ json);

        // function prototype to pass image processing parameters
        // int ProcessImage(string image, <params>);
    private:
}
#endif
