#ifndef SOM_H
#define SOM_H
#include <random>
#include <time.h>
#include <math.h>
#include "EasyBMP\EasyBMP.h"
#include <thrust\device_vector.h>
#define NUM_NEURONS 14400
using namespace std;
int numIterations;
float initLearningRate = .9;
int mapWidth = 120, mapHeight = 120;

struct Neuron{
	int X;
	int Y;
};
Neuron * d_neurons;
Neuron * neurons;
float * weights;
float * d_weights;
float * input_data;
float * d_input_data;
string int_to_str(int i){
	stringstream ss;
	ss << i;
	string str = ss.str();
	return str;
}
float map(float value,  float istart, float istop, float ostart, float ostop) {
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}
void readData(float * input_data){
	int numSamples = 40;
	string subject_id = "";
	string face_position_num = "";
	string face_num = "";
	string final = "";
	BMP image;
	int count = 0;
	int countImage = 0;
	for(int i = 1; i<=numSamples; i++){
		subject_id = "s"+int_to_str(i);
		for(int j = 1; j<=10; j++){
			face_position_num = int_to_str(j);
			for(int k = 0; k<3; k++){
				face_num = int_to_str(k);
				final = "..\\images\\"+subject_id+"\\"+face_position_num+"\\"+face_num+".bmp";
				image.ReadFromFile(final.c_str());
				//face_position is the orientation ID of the face

				for(int l = 0; l< 23; l++){
					for(int m = 0; m<28; m++){
						input_data[countImage] = map((((float)image(l, m)->Red)/255), 0, 1, -1, 1);
						countImage++;
					}
				}

			}
		}
	}

}
double calcDistBetweenNodes(Neuron n1, Neuron n2){

	double temp = (double)((n1.X-n2.X)*(n1.X-n2.X)+(n1.Y-n2.Y)*(n1.Y-n2.Y));
	return sqrt(temp);
}
double mapRadius(int time, int initialRadius, int newIterations){

	double initialMapRadius = (double)initialRadius;
	double timeConstant = newIterations/log(initialMapRadius);
	double radius = initialMapRadius*exp(-(time/timeConstant));
	return radius;
}	
double learningRate(int time, double newLearningRate, int newIterations){

	float iterations = (float)newIterations;
	double rate = newLearningRate*exp((-(time/iterations)));

	return rate;
}

float mapRadius(int time){

	double initialMapRadius = max(mapWidth, mapHeight)/1.5;
	double timeConstant = numIterations/log(initialMapRadius);
	double radius = initialMapRadius*exp(-(time/timeConstant));
	return radius;
}	

double learningRate(int time){
	float iterations = (float)numIterations;
	double rate = initLearningRate*exp((-(time/iterations)));

	return rate;
}
double theta(float distanceBetweenNodes, float radius){
	return exp(-(distanceBetweenNodes*distanceBetweenNodes)/(2*radius*radius));
}



string matlab1 = 
"c=zeros(70, 3)\n"
"for j=1:10,\n"
"    c(j, 1) = 0;\n"
"    c(j, 2) = 0;\n"
"    c(j, 3) = 1;\n"
"end\n"
"for j=11:20,\n"
"   c(j, 1) = 1;\n"
"  c(j, 2) = 0;\n"
"    c(j, 3) = 0;\n"
"end\n"
"for j=21:30,\n"
"    c(j, 1) = 0;\n"
"    c(j, 2) = 1;\n"
"    c(j, 3) = 0;\n"
"end\n"
"for j=31:40,\n"
"    c(j, 1) = 0;\n"
"    c(j, 2) = 0;\n"
"    c(j, 3) = 0;\n"
"end\n"
"for j=41:70,\n"
"    c(j, 1) = 1;\n"
"    c(j, 2) = 0;\n"
"    c(j, 3) = 1;\n"
"end\n";

string matlab2 = 

"figure\n" 
"grid on\n"

"hold on\n"
"fill( pointX0, pointY0, 'b')\n"
"alpha(0.3)\n"

"hold on\n"
"fill(pointX1, pointY1, 'r')\n"
"alpha(0.3)\n"
 
"hold on\n"
"fill(pointX2, pointY2, 'g')\n"
"alpha(0.3)\n"

"hold on\n"
"fill(pointX3, pointY3, 'black')\n"
"alpha(0.3)\n"


"h = scatter(X,Y, 70, c);\n"

"end\n";


#endif 