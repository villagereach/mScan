/*
 * include the headers required by the generated cpp code
 */
%{
using namespace cv;
%}

//import the android-cv.i file so that swig is aware of all that has been previous defined
//notice that it is not an include....
%import "android-cv.i"

//make sure to import the image_pool as it is 
//referenced by the Processor java generated
//class
%typemap(javaimports) ImageProcessor "
import com.opencv.jni.Mat;

/** ImageProcessor - for processing forms within mscan.
*/"
class ImageProcessing {
public:
    ImageProcessing();
    virtual ~ImageProcessing();
    int mscan();
};
