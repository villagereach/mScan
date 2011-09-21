/*
This program runs the image processing pipeline on every image in the specified folder (recusively)
then prints out stats breaking down the results by image label and pipeline stage.
*/
#include "Processor.h"
#include "FileUtils.h"
#include "StatCollector.h"
#include "MarkupForm.h"

#include <iostream>
#include <string>
#include <map>
#include <time.h>

using namespace std;

string getLabel(const string& filepath){
	int start = filepath.find_last_of("/") + 1;
	return	filepath.substr(start, filepath.find_last_of("_") - start);
}

int main(int argc, char *argv[]) {

	clock_t init, final;	
	
	map<string, StatCollector> collectors;

	string experimentDir(argv[1]);
	//Ensure the experiment dir ends with a '/'
	if(experimentDir[experimentDir.size() - 1] != '/') experimentDir.append("/");
	
	string inputDir("form_images/" + experimentDir);
	string outputDir("aligned_forms/" + experimentDir);
	string expectedJsonFile( "form_images/experiment.json" );

	string templatePath("form_templates/SIS-A01");

	vector<string> filenames;
	CrawlFileTree(inputDir, filenames);

	vector<string>::iterator it;
	for(it = filenames.begin(); it != filenames.end(); it++) {
		if( isImage((*it)) ){
		
			string label = getLabel(*it);
		
			collectors[label].incrImages();
			string relativePathMinusExt((*it).substr((*it).find_first_of(inputDir) + inputDir.length(),
																		(*it).length()-inputDir.length()-4));
			string imgOutputPath(outputDir + relativePathMinusExt + ".jpg");
			string markedupFormOutfile(outputDir + relativePathMinusExt + "_marked.jpg");
			string jsonOutfile(outputDir + relativePathMinusExt + ".json");
			
			cout << "Processing image: " << (*it) << endl;
			
			init = clock();
			
			Processor myProcessor("../assets/");
			
			if( !myProcessor.loadFormImage((*it).c_str(), true)) {
				cout << "\E[31m" <<  "Could not load. Arg: " << "\e[0m" << (*it) << endl;
				//Note that load errors could indicate a missing camera calibration file or a missing image.
				collectors[label].incrErrors();
				continue;
			}
			
			if( !myProcessor.loadFeatureData(templatePath.c_str()) ) {
				cout << "\E[31m" <<  "Could not set load feature data. Arg: " << "\e[0m" << templatePath << endl;
				continue;
			}
			
			if( !myProcessor.setTemplate(templatePath.c_str()) ) {
				cout << "\E[31m" <<  "Could not set template. Arg: " << "\e[0m" << templatePath << endl;
				continue;
			}
			
			cout << "Outputting aligned image to: " << imgOutputPath << endl;
			
			if( !myProcessor.alignForm(imgOutputPath.c_str()) ) {
				cout << "\E[31m" <<  "Could not align. Arg: " << "\e[0m" << imgOutputPath  << endl;
				collectors[label].incrErrors();
				continue;
			}
			
			if( !myProcessor.processForm(jsonOutfile.c_str()) ) {
				cout << "\E[31m" << "Could not process. Arg: " << "\e[0m" << jsonOutfile << endl;
				collectors[label].incrErrors();
				continue;
			}
			if( !MarkupForm::markupForm(jsonOutfile.c_str(), imgOutputPath.c_str(), markedupFormOutfile.c_str()) ) {
				cout << "\E[31m" <<  "Could not markup. Arg: " << "\e[0m" << markedupFormOutfile  << endl;
				collectors[label].incrErrors();
				continue;
			}
			
			final=clock()-init;
			cout << "Time taken: " << (double)final / ((double)CLOCKS_PER_SEC) << " seconds" << endl;
			cout << "\E[32m" << "Apparent success!" << "\e[0m" << endl;
			
			collectors[label].addTime( (double)final / ((double)CLOCKS_PER_SEC) );
			
			if(fileExists(jsonOutfile) && fileExists(expectedJsonFile) && fileExists(templatePath + ".json")){
				collectors[label].compareFiles(jsonOutfile, expectedJsonFile, COMP_BUBBLE_VALS);
				collectors[label].compareFiles(jsonOutfile, templatePath + ".json", COMP_BUBBLE_OFFSETS);
			}
			else{
				cout << "\E[31m" <<  "Could not compare files. File does not exist." << "\e[0m" << endl;
				continue;
			}
		}
	}
	
	//Print out overall stats:
	cout << linebreak << endl;
	StatCollector overall;
	map<string, StatCollector>::iterator mapit;
	for ( mapit=collectors.begin() ; mapit != collectors.end(); mapit++ ){
    	cout << (*mapit).first << endl << (*mapit).second;
    	overall += (*mapit).second;
	}
	cout << "overall" << endl << overall;
}
