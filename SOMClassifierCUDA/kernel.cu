// SOMClassifier.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include "EasyBMP\EasyBMP.h"
#include <time.h>
#include <thrust\extrema.h>
#include <math.h>
#include "SOM.h"
std::string get_file_contents(const char *filename)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return(contents);
	}
	throw(errno);
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
		positionNum = rand()%5;
		subjectNum = rand()%4;
		int count = 0;
		for(int i = (subjectNum*30+positionNum*3)*644; i<(subjectNum*30+positionNum*3)*644+644; i++){
			data[count] = input_data[i];
			count++;
		}
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);
		count = 0;
		findBMU<<<450, 32>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;

		neighboorhoodRadius = mapRadius(y);
		rate = learningRate(y);
		for(int h = 0; h<mapWidth; h++){
			for(int i = 0; i<mapHeight; i++){


				distance = calcDistBetweenNodes(neurons[ (int)(h*mapWidth+i) ], neurons[winner]);

				if(distance<neighboorhoodRadius){
					coeff = theta(distance, neighboorhoodRadius)*rate;
					float * newWeight;
					newWeight = new float [644];
					for (int w = 0; w<644; w++){
						double diff = data[w]-weights[(( (int)(h*mapWidth+i) )*644)+w];
						newWeight[w] =diff*coeff;
					}
					for (int w = 0; w<644; w++){
						weights[(( (int)(h*mapWidth+i) )*644)+w]+=newWeight[w];
					}
					delete newWeight;
				}

			}
		}
		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);
		if(y%10 == 0){
			cout << y << endl;
		}

	}
	free(data);
	free(distances);
	cudaFree(d_distances);
	cudaFree(d_data);
	free(winningNeuronID);
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
		positionNum = rand()%5;
		subjectNum = rand()%4;


		for(int i = (subjectNum*30+positionNum*3)*644; i<(subjectNum*30+positionNum*3)*644+644; i++){
			data[count] = input_data[i];
			count++;
		}
		count = 0;
		cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);
		findBMU<<<450, 32>>>(d_data, d_weights, d_distances);
		cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);

		winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
		winner = winnerID-distances;
		neighboorhoodRadius = mapRadius(y, initialSize,number_of_iterations );
		rate = learningRate(y, newLearningRate, number_of_iterations);



		for(int h = 0; h<mapWidth; h++){
			for(int i = 0; i<mapHeight; i++){


				distance = calcDistBetweenNodes(neurons[ (int)(h*mapWidth+i) ], neurons[winner]);

				if(distance<neighboorhoodRadius){
					coeff = theta(distance, neighboorhoodRadius)*rate;
					float * newWeight;
					newWeight = new float [644];
					for (int w = 0; w<644; w++){
						double diff = data[w]-weights[(( (int)(h*mapWidth+i) )*644)+w];
						newWeight[w] =diff*coeff;
					}
					for (int w = 0; w<644; w++){
						weights[(( (int)(h*mapWidth+i) )*644)+w]+=newWeight[w];
					}
					delete newWeight;
				}
			}
		}

		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);
		if(y%10 == 0){
			cout << y << endl;
		}
	}
	free(data);
	free(distances);
	cudaFree(d_distances);
	cudaFree(d_data);
}

void setXYZ(Neuron * neurons){

	for(int i = 0; i<mapHeight; i++){
		for(int j = 0; j<mapWidth; j++){	

			neurons[(int)(i*mapWidth+j)].X = i;
			neurons[(int)(i*mapWidth+j)].Y = j;

		}
	}
}

void writeWeights(const char * filename){
	ofstream file(filename);

	file.write( (const char *)(weights), 644*NUM_NEURONS*sizeof(float));
	file.close();
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
	int z;
	printf("Input number of epochs: ");
	cin >> z;

	printf("\nInitializing Neurons....");
	/*
	* SET XYZ FOR THE HOST AND DEVICE NEURONS
	*/
	//-------------------------------------------------
	neurons = (Neuron *)malloc(NUM_NEURONS*sizeof(Neuron)); // allocate memory for host neurons
	cudaMalloc((void**)&d_neurons, NUM_NEURONS*sizeof(Neuron)); // allocate memory for device neurons
	setXYZ(neurons); //set XYZ params on DEVICE side
	//-------------------------------------------------

	printf("complete \n");
	printf("Intializing Weights......");
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
	printf("complete \n");
	printf("Reading image data.....");
	/*
	* Read data from file
	*/
	//-------------------------------------------------
	input_data = (float *) malloc(1200*644*sizeof(float)); //allocate mem for host image data
	cudaMalloc(&d_input_data, 1200*644*sizeof(float)); //allocate mem for device image data
	readData(input_data); // read data with host array
	cudaMemcpy(d_input_data, input_data, 1200*644*sizeof(float), cudaMemcpyHostToDevice); //copy to device
	//-------------------------------------------------
	printf("complete \n");

	printf("Training started \n");

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
		
		numIterations = z;
		train();
		train(mapWidth/5, .2, numIterations/2);
		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice);

		int count = 0;



		ofstream file;
		string filename;

		filename = "Outputs\\data"+int_to_str(z)+"_test.txt";
		file.open(filename.c_str());
		for(int i = 0; i<210; i+=3){


			for(int j = i*644; j<i*644+644; j++){
				data[count] = input_data[j];
				count++;
			}
			cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);

			count = 0;
			findBMU<<<450, 32>>>(d_data, d_weights, d_distances);
			cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
			winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
			winner = winnerID-distances;
			file <<  neurons[winner].X<<(i==207 ? "" : ",");

		}
		file << endl;

		for(int i = 0; i<210; i+=3){

			for(int j = i*644; j<i*644+644; j++){
				data[count] = input_data[j];
				count++;
			}
			cudaMemcpy(d_data, data, 644*sizeof(float), cudaMemcpyHostToDevice);

			count = 0;
			findBMU<<<450, 32>>>(d_data, d_weights, d_distances);
			cudaMemcpy(distances, d_distances, sizeof(float)*NUM_NEURONS, cudaMemcpyDeviceToHost);
			winnerID = thrust::min_element(distances, distances+NUM_NEURONS);
			winner = winnerID-distances;
			file <<  neurons[winner].Y<<(i==207 ? "" : ",");

		}
		file << endl;
		file.close();
		filename = "java -jar ConvexHull.jar "+int_to_str(z)+"_test";
		system(filename.c_str());


		filename = "Weights\\weights"+int_to_str(z)+".bin";

		writeWeights(filename.c_str());
		setWeights(weights); // set weights on the HOST side
		cudaMemcpy(d_weights, weights, NUM_NEURONS*644*sizeof(float), cudaMemcpyHostToDevice); // copy over to device

		filename = "Outputs\\data"+int_to_str(z)+"_test.txt";
		std::ifstream t;
		t.open(filename.c_str());
		std::string buffer;
		std::string line;
		
		std::getline(t, line);
		line = "X = ["+line+"]";
		string x_and_y = line+'\n';
		std::getline(t, line);
		line = "Y = ["+line+"]";
		x_and_y += line+'\n';
		t.close();

		filename = "Outputs\\OutputVerts"+int_to_str(z)+"_test.txt";
		string points = get_file_contents(filename.c_str());

		string final_program = "function Display"+int_to_str(z)+'\n'+x_and_y+matlab1+points+matlab2;
		
		filename = "Matlab\\Display"+int_to_str(z)+".m";
		file.open(filename.c_str());

		file << final_program;
		file.close();
	
	free(data);
	free(weights);
	free(neurons);
	free(distances);
	free(input_data);

	cudaFree(d_data);
	cudaFree(d_weights);
	cudaFree(d_distances);
	cudaFree(d_input_data);
	cudaFree(d_neurons);
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