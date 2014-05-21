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
#include <thrust\extrema.h>
#include <math.h>
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
float initLearningRate = .8;
int mapWidth = 25, mapHeight = 25, mapDepth = 25;
//int initialMapRadius = 5;



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


__global__ void findBMU(float * inputVector, float * weights, float * distances){

	int i = threadIdx.x+(blockIdx.x*blockDim.x);

	if(i<NUM_NEURONS){

		int offset = i*644;
		int count = 0;
		float currentDistance = 0;
		for(int w = offset; w<offset+644; w++){
			currentDistance += abs((inputVector[count]-weights[w]))*abs((inputVector[count]-weights[w]));

			count++;
		}

		distances[i] = sqrt(currentDistance);		

	}

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
__global__ void printWeights(float*weights){
	int i = threadIdx.x+(blockIdx.x*blockDim.x);
	printf("%f\n", weights[i]);
}


void train(){
	Neuron winningNeuron;
	int *winningNeuronID;
	winningNeuronID = (int*)malloc(sizeof(int));
	int subjectNum;
	int positionNum;

	float neighboorhoodRadius;

	float rate;
	float distance;
	double coeff;

	float * data;
	data = (float*)malloc(sizeof(float)*644);
	float * d_data;
	cudaMalloc(&d_data, sizeof(float)*644);

	float * distances;
	distances = (float*)malloc(sizeof(float)*NUM_NEURONS);

	float * d_distances;
	cudaMalloc(&d_distances, sizeof(float)*NUM_NEURONS);

	int winner;
	float * winnerID;
	for(int y = 0; y<numIterations; y++){

		//select a random image
		positionNum = rand()%10;
		subjectNum = rand()%4;
		int count = 0;
		for(int i = (subjectNum*30+positionNum*3)*644; i<(subjectNum*30+positionNum*3)*644+644; i++){
			data[count] = input_data[i];
			count++;
		}
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);
		count = 0;
		findBMU<<<20, 50>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;

		neighboorhoodRadius = mapRadius(y);
		rate = learningRate(y);
		for(int h = 0; h<mapWidth; h++){
			for(int i = 0; i<mapHeight; i++){
				for(int j = 0; j<mapDepth; j++){

					distance = calcDistBetweenNodes(neurons[h*(int)(pow((double)mapWidth, 2.0)+i*(mapWidth)+j], neurons[winner]);

					if(distance<neighboorhoodRadius){
						coeff = theta(distance, neighboorhoodRadius)*rate;
						float * newWeight;
						newWeight = new float [644];
						for (int w = 0; w<644; w++){
							double diff = data[w]-weights[((h*100+i*10+j)*644)+w];
							newWeight[w] =diff*coeff;
						}
						for (int w = 0; w<644; w++){
							weights[((h*100+i*10+j)*644)+w]+=newWeight[w];
						}
						delete newWeight;
					}
				}
			}
		}
		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);
		if(y%10 == 0){
			cout << y << endl;
		}

	}
}


void train(int initialSize, float newLearningRate, float number_of_iterations){
	Neuron winningNeuron;
	int *winningNeuronID;
	winningNeuronID = (int*)malloc(sizeof(int));
	int subjectNum;
	int positionNum;

	float neighboorhoodRadius;

	float rate;
	float distance;
	double coeff;

	float * data;
	data = (float*)malloc(sizeof(float)*644);
	float * d_data;
	cudaMalloc(&d_data, sizeof(float)*644);

	float * distances;
	distances = (float*)malloc(sizeof(float)*NUM_NEURONS);

	float * d_distances;
	cudaMalloc(&d_distances, sizeof(float)*NUM_NEURONS);
	int count = 0;
	int winner;
	float * winnerID;

	for(int y = 0; y<number_of_iterations; y++){

	//select a random image
		positionNum = rand()%10;
		subjectNum = rand()%4;

		
		for(int i = (subjectNum*30+positionNum*3)*644; i<(subjectNum*30+positionNum*3)*644+644; i++){
			data[count] = input_data[i];
			count++;
		}
		count = 0;
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);
		findBMU<<<20, 50>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);

		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;
		neighboorhoodRadius = mapRadius(y, initialSize,number_of_iterations );
		rate = learningRate(y, newLearningRate, number_of_iterations);
		


		for(int h = 0; h<mapWidth; h++){
			for(int i = 0; i<mapHeight; i++){
				for(int j = 0; j<mapDepth; j++){

					distance = calcDistBetweenNodes(neurons[h*100+i*10+j], neurons[winner]);

					if(distance<neighboorhoodRadius){
						coeff = theta(distance, neighboorhoodRadius)*rate;
						float * newWeight;
						newWeight = new float [644];
						for (int w = 0; w<644; w++){
							double diff = data[w]-weights[((h*100+i*10+j)*644)+w];
							newWeight[w] =diff*coeff;
						}
						for (int w = 0; w<644; w++){
							weights[((h*100+i*10+j)*644)+w]+=newWeight[w];
						}
						delete newWeight;
					}
				}
			}
		}
		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);
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
void outputImageFromNetwork(float*input_data, int offset, char*outputFileName){
	BMP image;
	image.SetSize(23, 38);
	RGBApixel pixel;
	int count = 0;

	for(int i = 0; i<23; i++){
		for(int j = 0; j<28; j++){
			pixel.Red = map(input_data[offset+count], 0, 1, 0, 255);
			pixel.Blue = map(input_data[offset+count], 0, 1, 0, 255);
			pixel.Green = map(input_data[offset+count], 0, 1, 0, 255);
			image.SetPixel(i, j, pixel);
			count++;
		}
	}
	image.WriteToFile(outputFileName);

}


int main(int argc, char*argv[])
{
	printf("helllooooooo \n");
	/*
	* SET XYZ FOR THE HOST AND DEVICE NEURONS
	*/
	//-------------------------------------------------
	neurons = (Neuron *)malloc(NUM_NEURONS*sizeof(Neuron)); // allocate memory for host neurons
	cudaMalloc((void**)&d_neurons, NUM_NEURONS*sizeof(Neuron)); // allocate memory for device neurons
	dim3 dim = *(new dim3(10, 10, 10)); 
	setXYZ(neurons); //set XYZ params on DEVICE side
	//cudaMemcpy(neurons, d_neurons, NUM_NEURONS*sizeof(Neuron), cudaMemcpyDeviceToHost); // copy over to host
	//-------------------------------------------------
	printf("Stage 1 complete \n");
	/*
	* Initialize weights
	*/
	//------------------------------------------------
	srand(time(NULL));
	weights = (float*)malloc(NUM_NEURONS*644*sizeof(float)); // allocate mem for host weights
	cudaMalloc(&d_weights, NUM_NEURONS*644*sizeof(float)); // allocate mem for device weights
	setWeights(weights); // set weights on the HOST side
	cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice); // copy over to device
	//-------------------------------------------------
	printf("Stage 2 complete \n");
	/*
	* Read data from file
	*/
	//-------------------------------------------------
	input_data = (float *) malloc(1200*644*sizeof(float)); //allocate mem for host image data
	cudaMalloc(&d_input_data, 1200*644*sizeof(float)); //allocate mem for device image data
	readData(input_data); // read data with host array
	cudaMemcpy(d_input_data, input_data, 1200*644*sizeof(float), cudaMemcpyHostToDevice); //copy to device
	//-------------------------------------------------

	printf("Training started \n");
	train();
	train(2, .02, 500);
	cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);

	int count = 0;

	float * data;
	data = (float*)malloc(sizeof(float)*644);
	float * d_data;
	cudaMalloc(&d_data, sizeof(float)*644);

	float * distances;
	distances = (float*)malloc(sizeof(float)*NUM_NEURONS);
	float * d_distances;
	cudaMalloc(&d_distances, sizeof(float)*NUM_NEURONS);

	int winner;
	float *winnerID;

	
	ofstream file;
	file.open("data.txt");
	for(int i = 0; i<120; i+=3){
		for(int j = i*644; j<i*644+644; j++){
			data[count] = input_data[j];
			count++;
		}
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);

		count = 0;
		findBMU<<<20, 50>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;
		file <<  neurons[winner].X<<(i==117 ? "" : ",");
	}
	file << endl;

	for(int i = 0; i<120; i+=3){
		for(int j = i*644; j<i*644+644; j++){
			data[count] = input_data[j];
			count++;
		}
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);

		count = 0;
		findBMU<<<20, 50>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;
		file <<  neurons[winner].Y<<(i==117 ? "" : ",");
	}
	file << endl;

	for(int i = 0; i<120; i+=3){
		for(int j = i*644; j<i*644+644; j++){
			data[count] = input_data[j];
			count++;
		}
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);

		count = 0;
		findBMU<<<20, 50>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;
		file <<  neurons[winner].Z<<(i==117 ? "" : ",");
	}
	
	file.close();

	outputImageFromNetwork(weights, 0, "OutputFileFromNetworkWeights.bmp");
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