#ifndef SOM_H
#define SOM_H
#include <random>
#include <time.h>
#include <math.h>
#include "EasyBMP\EasyBMP.h"
using namespace std;
class neuron{
public:
	float * weights;
	int numWeights;
	int X;
	int Y;
	int Z;
	neuron(){
		numWeights = 0;

	}
	void initWeights(int number_of_weights){
		numWeights = number_of_weights;


		weights = new float[numWeights];
		for (int i = 0; i<numWeights; i++){
			weights[i] = ((float)rand()/RAND_MAX);
		}
	}
	void initWeights(int number_of_weights, float * weights_input){
		numWeights = number_of_weights;
		weights = new float[number_of_weights];
		for (int i = 0; i<number_of_weights; i++){
			weights[i] = weights_input[i];
		}

	}
};

class SOM{
private:

public:
	neuron *** neurons;

	int weight_dim;
	float *input;
	int mapWidth;
	int mapDepth;
	int mapHeight;
	int numIterations;
	float initLearningRate;
	struct image_data{
		int subject_num;
		int face_position;
		int face_num;
		float * data;
	};
	image_data *input_data;
	//Utility function to convert int to string
	string int_to_str(int i){
		stringstream ss;
		ss << i;
		string str = ss.str();
		return str;
	}
	float map(float value,  float istart, float istop, float ostart, float ostop) {
		return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
	}
	/*
		int wid -- width of map
		int len -- length of map
		int dep -- depth of map
		int dim -- the length of the feature vector
		int num -- the number of iterations
	*/
	SOM(int wid, int len, int dep, int dim, int num, float rate){
		srand(time(NULL));
		mapWidth = wid;
		mapHeight = len;
		mapDepth = dep;
		weight_dim = dim;
		numIterations = num;
		initLearningRate = rate;
		neurons = new neuron**[mapWidth];
		for(int x = 0; x<mapWidth; x++){
			neurons[x] = new neuron*[mapHeight];
		}
		for (int x = 0; x<mapWidth; x++){
			for(int y = 0; y<mapHeight; y++){
				neurons[x][y] = new neuron[mapDepth];
			}
		}
		for(int x = 0; x<mapWidth; x++){
			for (int y = 0; y<mapHeight; y++){
				for(int z = 0; z<mapDepth; z++){
					float *weights_input;
					weights_input = new float[weight_dim];
					for(int i = 0; i< weight_dim; i++){
						weights_input[i] = (float)rand()/RAND_MAX;


					}
					srand(time(NULL));
					neurons[x][y][z].initWeights(weight_dim);
					neurons[x][y][z].X = x;
					neurons[x][y][z].Y = y;
					neurons[x][y][z].Z = z;
				}
			}
		}
		input = new float[dim];
	}


	void readData(int numSamples){
		input_data = new image_data[numSamples*400*3];
		string subject_id = "";
		string face_position_num = "";
		string face_num = "";
		string final = "";
		BMP image;
		int count = 0;
		for(int i = 1; i<=numSamples; i++){
			subject_id = "s"+int_to_str(i);

			for(int j = 1; j<=10; j++){
				face_position_num = int_to_str(j);
				for(int k = 0; k<3; k++){
					face_num = int_to_str(k);
					final = "..\\images\\"+subject_id+"\\"+face_position_num+"\\"+face_num+".bmp";
					image.ReadFromFile(final.c_str());
					//face_position is the orientation ID of the face
					input_data[count].face_position = j;
					//face_num is which of the 3 faces it is
					input_data[count].face_num = k;
					input_data[count].subject_num = i;
					input_data[count].data = new float[644];
					int countImage = 0;
					for(int l = 0; l< 23; l++){
						for(int m = 0; m<28; m++){
							input_data[count].data[countImage] = map((((float)image(l, m)->Red)/255), 0, 1, -1, 1);
							countImage++;
						}
					}
					count++;
				}
			}
		}

	}
	void initInputVector(float * vector){
		for (int i = 0; i<weight_dim; i++){
			input[i] = vector[i];
		}
	}

	float calcDist(float *vec1, float *vec2){
		float distance = 0.0f;
		for (int i = 0; i<weight_dim; i++){

			distance += (abs(vec1[i]-vec2[i]))*(abs(vec1[i]-vec2[i]));
		}
		return sqrt(distance);
	}

	double calcDistBetweenNodes(neuron n1, neuron n2){

		double temp = (double)((n1.X-n2.X)*(n1.X-n2.X)+(n1.Y-n2.Y)*(n1.Y-n2.Y)+(n1.Z-n2.Z)*(n1.Z-n2.Z));
		return sqrt(temp);
	}




	float mapRadius(int time){

		double initialMapRadius = max(mapWidth, mapHeight)/1.5;
		double timeConstant = numIterations/log(initialMapRadius);
		double radius = initialMapRadius*exp(-(time/timeConstant));
		return radius;
	}	

	double mapRadius(int time, int initialRadius, int newIterations){

		double initialMapRadius = (double)initialRadius;
		double timeConstant = newIterations/log(initialMapRadius);
		double radius = initialMapRadius*exp(-(time/timeConstant));
		return radius;
	}	

	neuron findBMU(float * data1){
		float leastDistance = 9999.0f;
		float currentDistance = 0.0f;
		neuron winner;
		for(int h = 0; h<mapWidth; h++){
			for (int i = 0; i<mapHeight; i++){
				for(int j = 0; j<mapDepth; j++){
					currentDistance = calcDist(neurons[h][i][j].weights, data1);

					if(currentDistance<leastDistance){
						winner = neurons[h][i][j];

						leastDistance = currentDistance;

					}
				}
			}
		}
		return winner;
	}


	double theta(float distanceBetweenNodes, float radius){
		return exp(-(distanceBetweenNodes*distanceBetweenNodes)/(2*radius*radius));
	}

	double learningRate(int time, double newLearningRate, int newIterations){

		float iterations = (float)newIterations;
		double rate = newLearningRate*exp((-(time/iterations)));

		return rate;
	}
	double learningRate(int time){
		float iterations = (float)numIterations;
		double rate = initLearningRate*exp((-(time/iterations)));

		return rate;
	}

	void train(){
		neuron winningNeuron;
		
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

			winningNeuron = findBMU(input_data[subjectNum*30+positionNum*3].data);
			winX = winningNeuron.X;
			winY = winningNeuron.Y;
			winZ = winningNeuron.Z;
			neighboorhoodRadius = mapRadius(y);
			rate = learningRate(y);
			for(int h = 0; h<mapWidth; h++){
				for(int i = 0; i<mapHeight; i++){
					for(int j = 0; j<mapDepth; j++){

						distance = calcDistBetweenNodes(neurons[h][i][j], neurons[winX][winY][winZ]);
						if(distance<neighboorhoodRadius){
							coeff = theta(distance, neighboorhoodRadius)*rate;
							float * newWeight;
							newWeight = new float [weight_dim];
							for (int w = 0; w<weight_dim; w++){
								double diff = input_data[subjectNum*30+positionNum*3].data[w]-neurons[h][i][j].weights[w];
								newWeight[w] =diff*coeff;
							}
							for (int w = 0; w<weight_dim; w++){
								neurons[h][i][j].weights[w]+=newWeight[w];
							}
							delete newWeight;
						}
					}
				}
			}
			if(y%10 == 0){
				cout << y << endl;
			}
		}



	}

	void train(int initialSize, int number_of_iterations, double newLearningRate){
		neuron winningNeuron;
		
		int subjectNum;
		int positionNum;

		int winX;
		int winY;
		int winZ;
		float neighboorhoodRadius;
		
		float rate;
		float distance;
		double coeff;
		
		for(int y = 0; y<number_of_iterations; y++){

			//select a random image
			positionNum = rand()%10;
			subjectNum = rand()%4;

			winningNeuron = findBMU(input_data[subjectNum*30+positionNum*3].data);
			winX = winningNeuron.X;
			winY = winningNeuron.Y;
			winZ = winningNeuron.Z;
			neighboorhoodRadius = mapRadius(y, initialSize,number_of_iterations );
			 rate = learningRate(y, newLearningRate, number_of_iterations);
			for(int h = 0; h<mapWidth; h++){
				for(int i = 0; i<mapHeight; i++){
					for(int j = 0; j<mapDepth; j++){

						distance = calcDistBetweenNodes(neurons[h][i][j], neurons[winX][winY][winZ]);
						if(distance<neighboorhoodRadius){
							coeff = theta(distance, neighboorhoodRadius)*rate;
							float * newWeight;
							newWeight = new float [weight_dim];
							for (int w = 0; w<weight_dim; w++){
								double diff = input_data[subjectNum*30+positionNum*3].data[w]-neurons[h][i][j].weights[w];
								newWeight[w] =diff*coeff;
							}
							for (int w = 0; w<weight_dim; w++){
								neurons[h][i][j].weights[w]+=newWeight[w];
							}
							delete newWeight;
						}
					}
				}
			}
			if(y%10 == 0){
				cout << y << endl;
			}
		}



	}
	
	



}; 

#endif 