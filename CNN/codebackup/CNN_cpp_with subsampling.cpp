// CNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNN.h"
using namespace std;
int main(int argc, char* argv[])
{
	srand(time(NULL));
	CNN c("config.conf", 6);
	BMP bitmap;
	bitmap.ReadFromFile("0.bmp");
	int count = 0;
	for(int i = 0; i<23; i++){
		for(int j = 0; j<28; j++){
			c.layers[0].neurons[count].output = (float)bitmap(i, j)->Red/255;
			
			count++;
		}
	}

	cout << endl;
	c.propagate();	
	cout << "Before Training -> "<<c.layers[c.numLayers-1].neurons[4].output << endl;
	
	for(int i = 0; i<1; i++){
	c.backprop(1);
	c.propagate();
	}
	c.propagate();
	cout << "0->" <<c.layers[c.numLayers-1].neurons[0].output << endl;
	for(int i = 0; i<1; i++){
	c.backprop(0);
	c.propagate();
	}
	c.propagate();
	cout << "-----------" << endl;
	cout << "1->"<<c.layers[c.numLayers-1].neurons[0].output << endl;
	c.readData(40);
	

	
	//cout << c.layers[3].neurons[0].output << endl;
	return 0;
}

 
