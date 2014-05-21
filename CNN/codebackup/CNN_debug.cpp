// CNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNN.h"
#include <iomanip>
using namespace std;
int main(int argc, char* argv[])
{
	srand(time(NULL));
	CNN c("config.conf", 6);
	ofstream fout;
	BMP bitmap;
	int count = 0;	
	fout.open("weights_before_anything.txt");
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


	cout << "strt" << endl;
	c.readData(40);
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
	count = 0;
	fout.open("before_training_output.txt");
		for(int z = 0; z<12; z+=3){



		c.propagate();
		fout << z << " ";
		for(int i =0 ; i<40; i++){
			fout << c.layers[5].neurons[i].output << " ";
		}

		fout << endl;
		count = 0;
	}

		fout.close();

	fout.open("weights_after_one_propagate.txt");
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


	c.train(1);
	count = 0;
	fout.open("inputweightid.txt");
	for(int i =1; i<c.numLayers; i++){
		for(int j = 0; j<c.layers[i].numNeurons; j++){
			for(int k = 0; k<c.layers[i].neurons[j].inputWeightID.size(); k++){
				fout <<c.layers[i].neurons[j].inputWeightID[k] << " ";
			}
			fout << endl;
		}
		fout << "---" << endl;
	}

	fout.close();
	fout.open("inputid.txt");
	for(int i =1; i<c.numLayers; i++){
		for(int j = 0; j<c.layers[i].numNeurons; j++){
			fout << j<<"-";
			for(int k = 0; k<c.layers[i].neurons[j].inputs.size(); k++){
				fout <<c.layers[i].neurons[j].inputs[k] << " ";
			}
			fout << endl;
		}
		fout << "---" << endl;
	}
	fout.close();
	fout.open("output1.txt");
	ofstream weights;
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
	for(int z = 0; z<12; z+=3){

		fout.precision(9);

		c.propagate();
		fout << z << " ";
		for(int i =0 ; i<1; i++){
			fout << c.layers[5].neurons[i].output << " ";
		}

		fout << endl;
		count = 0;
	}



	fout.close();
	fout.open("weights_after_training.txt");
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
	return 0;

}

