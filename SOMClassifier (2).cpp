// SOMClassifier.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "EasyBMP\EasyBMP.h"
#include <time.h>

#include <math.h>
#include "stdafx.h"
#define NUM_NEURONS 1000
struct Neuron{
	int X;
	int Y;
	int Z;
};


Neuron * d_neurons;
Neuron * neurons;
float * weights;
float * d_weights;
float * input_data;
float * d_input_data;

using namespace std;
int numIterations = 1000;
float initLearningRate = .9;
int mapWidth = 10, mapHeight = 10, mapDepth = 10;
int initialMapRadius = 5;



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

	double temp = (double)((n1.X-n2.X)*(n1.X-n2.X)+(n1.Y-n2.Y)*(n1.Y-n2.Y)+(n1.Z-n2.Z)*(n1.Z-n2.Z));
	return sqrt(temp);
}


int findBMU(float * inputVector, float * weights){


	int count = 0;
	float currentDistance = 0;
	int winner = 0;
	float leastDistance = 99999;
	//if(i<10&&j<10&&k<10){
	for(int i = 0; i<10; i++){
		for(int j = 0;j<10; j++){
			for(int k = 0; k<10; k++){
				int offset = (i*100+j*10+k)*644;
				for(int i = offset; i<offset+644; i++){
					currentDistance = (inputVector[count]-weights[i])*(inputVector[count]-weights[i]);
					count++;
				}
				count = 0;
				if(currentDistance<leastDistance){
					winner = offset;

					leastDistance = currentDistance;

				}
			}
			//}

		}
	}
	return winner;
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

void train(float *weights, Neuron*neurons, float*input_data){
	Neuron winningNeuron;
	int winningNeuronID = 0;
	int subjectNum;
	int positionNum;

	int winX;
	int winY;
	int winZ;
	float neighboorhoodRadius;

	float rate;
	float distance;
	double coeff;

	for(int y = 0; y<numIterations; y++){

		//select a random image
		positionNum = rand()%10;
		subjectNum = rand()%4;
		float * data = new float[644];
		int count = 0;
		for(int i = (rand()%1200)*644; i<(rand()%1200)*644+644; i++){
			data[count] = input_data[i];
			count++;
		}
		
		findBMU(data, weights);
		
		winningNeuronID = winningNeuronID/644;
		neighboorhoodRadius = mapRadius(y);
		rate = learningRate(y);
		winX = (winningNeuronID/644)/100;
		winY = ((winningNeuronID/644)-(winX*100))*10;
		winZ = ((winningNeuronID/644)-(winX*100)-(winY*10));


		printf("1...");
		for(int h = 0; h<mapWidth; h++){
			for(int i = 0; i<mapHeight; i++){
				for(int j = 0; j<mapDepth; j++){

					distance = calcDistBetweenNodes(neurons[h*100+i*10+j], neurons[winningNeuronID/644]);

					if(distance<neighboorhoodRadius){
						coeff = theta(distance, neighboorhoodRadius)*rate;
						float * newWeight;
						newWeight = new float [644];
						for (int w = 0; w<644; w++){
							double diff = weights[(h*100+i*10+j)*644+w];
							newWeight[w] =diff*coeff;
						}
						for (int w = 0; w<644; w++){
							weights[(h*100+i*10+j)*644+w]+=newWeight[w];
						}
						delete newWeight;
					}
				}
			}
		}
		printf("2 \n");
		if(y%10 == 0){
			cout << y << endl;
		}
	}



}

void setXYZ(Neuron * neurons){

	for(int i = 0; i<10; i++){
		for(int j = 0; j<10; j++){
			for(int k = 0; k<10; k++){
				neurons[i*100+j*10+k].X = i;
				neurons[i*100+j*10+k].Y = j;
				neurons[i*100+j*10+k].Z = k;

			}
		}
	}
}

//this is done on HOST side because rand() can't be called device side
void setWeights(float * weights){
	for(int i = 0; i<NUM_NEURONS*644; i++){
		weights[i] = (double)rand()/RAND_MAX;
	}
}

int main(int argc, char*argv[])
{

	
	printf("asdf");
	neurons = new Neuron[1000];
	printf("asdf");
	setXYZ(neurons);
	printf("Stage 1 complete \n");
	
	srand(time(NULL));
	weights = new float[NUM_NEURONS*644]; 
	setWeights(weights); 
	
	printf("Stage 2 complete \n");

	input_data = new float[1200*644]; 
	readData(input_data); // read data with host array
	

	printf("Training started \n");
	

	train(weights, neurons, input_data);


}

/*
float * test_vector;
test_vector = (float*)malloc(644*sizeof(float));
for(int i = 0; i<644; i++){
test_vector[i] = .32f;
}
float * d_test_vector;
cudaMalloc(&d_test_vector, 644*sizeof(float));
cudaMemcpy(d_test_vector, test_vector, 644*sizeof(float), cudaMemcpyHostToDevice);
float * d_least;
float * least;
least = (float*)malloc(sizeof(float));
*least = 9999999;
cudaMalloc(&d_least, sizeof(float));
cudaMemcpy(d_least, least, sizeof(float), cudaMemcpyHostToDevice);

int * d_winner;
int * winner;
winner = (int*)malloc(sizeof(int));
*winner = 0;
cudaMalloc(&d_winner, sizeof(int));
cudaMemcpy(d_winner, winner, sizeof(int), cudaMemcpyHostToDevice);
*/