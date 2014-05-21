// CNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNN.h"
#include <iomanip>
using namespace std;

void outputWeights(CNN * c, char* name){
	ofstream fout;
	fout.open(name);
	for(int z = 1; z<c->numLayers-2; z+=2){
		for(int i = 0; i<c->layers[z].numOfMaps; i++){
			for(int j = 0; j<c->layers[z].mapSize; j++){
				for(int k = 0; k<c->layers[z].mapSize; k++){
					fout << c->layers[z].data[i].weights[j][k] << " ";
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
	for(int i =0 ; i<c->layers[c->numLayers - 1].numNeurons; i++){
		for(int j = 0; j<c->layers[c->numLayers - 2].numNeurons; j++){
			fout << c->layers[c->numLayers - 1].data[0].weights[i][j] << endl;
		}
	}
	fout.close();
}

void printOutputs(CNN * c){

	for(int i = 0; i<3; i++){
		cout << c->layers[c->numLayers - 1].neurons[i].output << endl;

	}

}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	CNN c("config.conf", 6);
	outputWeights(&c, "weights_before_anything.txt");
	BMP bitmap;
	c.readData(40);
	ofstream fout;
	c.loadImage(9);
	int count = 0;

	c.propagate();


	
	#pragma region print
	RGBApixel pixel;
	bitmap.SetSize(22, 27);

	fout.open("outputs_before.txt");
	c.loadImage(0);
	c.propagate();
	for(int i = 0; i<c.numLayers; i++){
		for(int j = 0; j<c.layers[i].numNeurons; j++){
			fout << c.layers[i].neurons[j].output << " ";
		}
		fout << endl;
	}


	fout << endl;
	c.loadImage(60);
	c.propagate();
	for(int i = 0; i<c.numLayers; i++){
		for(int j = 0; j<c.layers[i].numNeurons; j++){
			fout << c.layers[i].neurons[j].output << " ";
		}
		fout << endl;
	}

	fout.close();

	//	cout << c.layers[2].data[19].weights[0][0] << endl;

	for(int i = 0; i<22; i++){
		for(int j = 0; j<27; j++){
			pixel.Red = c.layers[1].neurons[count].output * 255;
			pixel.Green = c.layers[1].neurons[count].output * 255;
			pixel.Blue = c.layers[1].neurons[count].output * 255;
			pixel.Alpha = 0;
			count ++;
			bitmap.SetPixel(i, j, pixel);
		}
	}


	bitmap.WriteToFile("3rdlayeroutput.bmp");

#pragma endregion
	cout << "Hi!" << endl;
	c.layers[5].neurons[0].expected = 1;
	c.train(100);
	cout << endl;
	c.loadImage(0);
	c.propagate();
	printOutputs(&c);
	//c.testConnections(23, 28);
	outputWeights(&c, "weights_after_training.txt");
	count = 0;

	bitmap.SetSize(22, 27);
	for(int z = 0; z<c.layers[1].numOfMaps; z++){
		string filename = "";
		for(int i = 0; i<22; i++){
			for(int j = 0; j<27; j++){
				pixel.Red = c.layers[1].neurons[count].output * 255;
				pixel.Green = c.layers[1].neurons[count].output * 255;
				pixel.Blue = c.layers[1].neurons[count].output * 255;
				pixel.Alpha = 0;
				count ++;
				bitmap.SetPixel(i, j, pixel);
			}
		}
		filename = "intermediate_outputs\\layer_2_image"+c.int_to_str(z)+".bmp";
		bitmap.WriteToFile(filename.c_str());
	}
	count = 0;
	c.loadImage(30);
	c.propagate();
	bitmap.SetSize(22, 27);
	for(int z = 0; z<c.layers[1].numOfMaps; z++){
		string filename = "";
		for(int i = 0; i<22; i++){
			for(int j = 0; j<27; j++){
				pixel.Red = c.layers[1].neurons[count].output * 255;
				pixel.Green = c.layers[1].neurons[count].output * 255;
				pixel.Blue = c.layers[1].neurons[count].output * 255;
				pixel.Alpha = 0;
				count ++;
				bitmap.SetPixel(i, j, pixel);
			}
		}
		filename = "intermediate_outputs\\layer_2_image"+c.int_to_str(z+10)+".bmp";
		bitmap.WriteToFile(filename.c_str());
	}
	//ofstream fout;
	fout.open("TEST_OUTPUTS.txt");
	count = 0;
	for(int i =0; i<c.layers[2].numOfMaps; i++){
		for(int j = 0; j<c.layers[2].y_count; j++){
			for(int k = 0; k<c.layers[2].x_count; k++){
				fout << c.layers[2].neurons[count].output << " ";
				count ++;
			}
			fout << endl;
		}
		fout << endl;
	}
	fout << endl;

	for(int i =0; i<c.layers[3].numOfMaps; i++){
		for(int j = 0; j<c.layers[3].mapSize; j++){
			for(int k = 0; k<c.layers[3].mapSize; k++){
				fout << c.layers[3].data[i].weights[j][k] << " ";
			}
			fout << endl;
		}
		fout << endl;
	}
	fout << c.layers[3].data[0].bias << " <- bias" << endl;
	fout << c.layers[3].neurons[0].output << " <- output " <<endl;
	fout.close();
			fout.open("outputs_after.txt");
	for(int i = 0; i<60; i+=3){
		c.loadImage(i);
		c.propagate();
		fout << c.layers[5].neurons[0].output << endl;
	}

	count = 0;

	fout.close();
	system("pause");
	//cout << c.layers[2].data[0].weights[0][0] << " " <<  c.layers[2].data[1].weights[0][0]<< " hello" << endl;
	//cout << c.layers[3].neurons[0].output << endl;
	return 0;
}


/*
#pragma region printout
c.loadImage(6);
c.propagate();

fout.open("outputs_before.txt");
for(int i = 0; i<6; i++){
for(int j = 0; j<40; j++){
fout << c.layers[i].neurons[j].output << " ";
}
fout << endl;
}
fout << "--" << endl;

cout << endl;
c.loadImage(9);
c.propagate();
for(int i = 0; i<6; i++){
for(int j = 0; j<40; j++){
fout << c.layers[i].neurons[j].output << " ";
}
fout << endl;
}
fout << "--" << endl;

fout.close();
#pragma endregion 
c.loadImage(9);
c.layers[c.numLayers-1].neurons[0].expected = 1.0f;
for(int i = 1; i<40; i++){
c.layers[c.numLayers-1].neurons[i].expected = 0.0f;
}
for(int i = 0; i<10; i++){
c.backprop(0);
cout << i << endl;
}

c.propagate();
printOutputs(&c);
cout << "---" << endl;


c.loadImage(6);
c.layers[c.numLayers-1].neurons[0].expected = 0.0f;
c.layers[c.numLayers-1].neurons[1].expected = 1.0f;
for(int i = 0; i<10; i++){
c.backprop(0);
cout << i << endl;
}

#pragma region printout
c.loadImage(6);
c.propagate();
printOutputs(&c);

fout.open("outputs.txt");
for(int i = 0; i<6; i++){
for(int j = 0; j<40; j++){
fout << c.layers[i].neurons[j].output << " ";
}
fout << endl;
}
fout << "--" << endl;

cout << endl;
c.loadImage(9);
c.propagate();
for(int i = 0; i<6; i++){
for(int j = 0; j<40; j++){
fout << c.layers[i].neurons[j].output << " ";
}
fout << endl;
}
fout << "--" << endl;

fout.close();
#pragma endregion 
*/