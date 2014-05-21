// CNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNN.h"
#include <iomanip>
using namespace std;

void readimage(CNN &c){
	int count = 0;
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{
			c.layers[0].neurons[count].output = c.input_data[0].data[k][l];
			count++;
		}
	}
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{
			c.layers[0].neurons[count].output = c.input_data[1].data[k][l];
			count++;
		}
	}
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{

			c.layers[0].neurons[count].output = c.input_data[2].data[k][l];
			count++;
		}
	}
}

void readimage(CNN &c, int num){
	int count = 0;
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{
			c.layers[0].neurons[count].output = c.input_data[num].data[k][l];
			count++;
		}
	}
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{
			c.layers[0].neurons[count].output = c.input_data[num+1].data[k][l];
			count++;
		}
	}
	for(int k = 0; k<23; k++)
	{
		for(int l = 0; l<28; l++)
		{

			c.layers[0].neurons[count].output = c.input_data[num+2].data[k][l];
			count++;
		}
	}
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	CNN c("config.conf", 6);
	ofstream fout;
	int count = 0;

	c.readData(40);
	c.loadImage(0);

	fout.open("weights_before_anythin.txt");
	for(int z = 1; z<c.numLayers-2; z+=2){
		for(int i = 0; i<c.layers[z].numOfMaps; i++){
			for(int j = 0; j<c.layers[z].mapSize; j++){
				for(int k = 0; k<c.layers[z].mapSize; k++){
					fout << c.layers[z].data[i].weights[j][k] << " ";
				}
				fout << endl;
			}
			fout << endl;
			fout << endl;
		}
		fout << endl;
		fout << endl;
		fout <<"--------------------" << endl;
	}
	for(int i =0 ; i<c.layers[5].numNeurons; i++){
		for(int j = 0; j<c.layers[5].numNeurons; j++){
			fout << c.layers[5].data[0].weights[i][j] << endl;
		}
	}
	fout.close();
	fout.open("before_training.txt");
	fout.precision(9);	
	fout.setf(ios::fixed);
	fout.setf(ios::showpoint);

	for(int i = 0; i<60; i++){
		c.loadImage(i);
		c.propagate();
		for(int j = 0; j<c.layers[5].numNeurons; j++){
			fout << c.layers[5].neurons[j].output << " ";

		}
		c.reset();
		fout << "------" << endl;
		fout << endl;
	}
	fout.close();

	c.train(1);

	fout.open("after_training.txt");

	fout.precision(15);	
	fout.setf(ios::fixed);
	fout.setf(ios::showpoint);

	for(int i = 0; i<60; i+=3){
		c.loadImage(i);

		c.propagate();
		for(int j = 0; j<40; j++){
			fout << c.layers[5].neurons[j].output << " ";
		}

		fout << endl;
		c.reset();
	}
	fout.close();
	fout.open("weights.txt");
	for(int z = 1; z<c.numLayers-2; z+=2){
		for(int i = 0; i<c.layers[z].numOfMaps; i++){
			for(int j = 0; j<c.layers[z].mapSize; j++){
				for(int k = 0; k<c.layers[z].mapSize; k++){
					fout << c.layers[z].data[i].weights[j][k] << " ";
				}
				fout << endl;
			}
			fout << endl;
			fout << endl;
		}
		fout << endl;
		fout << endl;
		fout <<"--------------------" << endl;
	}
	for(int i =0 ; i<c.layers[5].numNeurons; i++){
		for(int j = 0; j<c.layers[4].numNeurons; j++){
			fout << c.layers[5].data[0].weights[i][j] << endl;
		}
	}

	fout.close();

	fout.open("bias.txt");
	for(int z = 1; z<c.numLayers-2; z+=2){
		for(int i = 0; i<c.layers[z].numNeurons; i++){

			fout << c.layers[z].neurons[i].bias << " ";
		}
		fout << endl;



		fout <<"--------------------" << endl;
	}
	return 0;

}

