// SOMClassifier.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SOM.h"

#include <iostream>

using namespace std;

void outputImageFromNetwork(int x, int y, int z, SOM * map, char * outputFileName){
	BMP image;
	image.SetSize(23, 38);
	RGBApixel pixel;
	int count = 0;

	for(int i = 0; i<23; i++){
		for(int j = 0; j<28; j++){
			pixel.Red = map->map(map->neurons[x][y][z].weights[count], 0, 1, 0, 255);
			pixel.Blue = map->map(map->neurons[x][y][z].weights[count], 0, 1, 0, 255);
			pixel.Green = map->map(map->neurons[x][y][z].weights[count], 0, 1, 0, 255);
			image.SetPixel(i, j, pixel);
			count++;
		}
	}
	image.WriteToFile(outputFileName);

}

void createPolygon(){
	ifstream file1;
	ofstream file;
	file1.open("dat.txt");
	int faces[28];
	float verts[9][3];
	int final_verts[28][3];

	for(int i = 0; i<9; i++){
		for(int j = 0; j<3; j++){
			file1 >> verts[i][j];
		}
	}

	
	for(int i = 0; i<28; i++){
			file1 >> faces[i];
		
	}


	for(int i = 0; i<28; i++){
		final_verts[i][0] = verts[faces[i]][0];
		final_verts[i][1] = verts[faces[i]][1];
		final_verts[i][2] = verts[faces[i]][2];
	}

	file1.close();
	file.open("outputdat.txt");
	for(int i = 0; i<28; i++){
		file << final_verts[i][0] << ",";
	}
	file << "\n";
	for(int i = 0; i<28; i++){
		file << final_verts[i][1] << ",";
	}
	file << "\n";
	for(int i = 0; i<28; i++){
		file << final_verts[i][2] << ",";
	}
	file.close();

	return;
}

int main(int argc, char*argv[])
{

	SOM map (10, 10, 10, 644, 10000, .8);

	/*Declarations
	numSubjects -- number of subjects in the test
	*/
	int numSubjects = 40;
	

	map.readData(numSubjects);

	map.train();
	map.train(4, 5000, .02);

	//output stream to file
	ofstream file;

	file.open("data.txt");
	cout << endl;
	for(int i = 0; i<120; i+=3){
		file <<  map.findBMU(map.input_data[i].data).X<<(i==117 ? "" : " ");
	}
	file << endl;
	for(int i = 0; i<120; i+=3){
		
		file <<  map.findBMU(map.input_data[i].data).Y<<(i==117 ? "" : " ");
	}
	file << endl;
	for(int i = 0; i<120; i+=3){
		
		file <<  map.findBMU(map.input_data[i].data).Z<<(i==117 ? "" : " ");
	}
	file.close();


	outputImageFromNetwork(0, 0, 0, &map, "OutputOfNetworkWeights.bmp");

	system("java -jar ConvexHull.jar");

	return 0;
}

