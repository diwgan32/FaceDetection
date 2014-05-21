// SOM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
using namespace std;




void write_to_file(SOM network, float ** data){


}

string int_to_str(int i){
	stringstream ss;
	ss << i;
	string str = ss.str();
	return str;
}

struct boxPos{
	int startX;
	int startY;
	int endX;
	int endY;
};
int main(int argc, char* argv[])
{

	SOM network(5, 5, 5, 25);

	BMP *output;

	BMP input;

	int imageX = 92;
	int imageY = 112;
	boxPos pos;
	pos.endX = 0;
	pos.endY = 0;
	int numIterationsCompletedCount = 0;
	pos.startX = 0;
	pos.startY = 0;
	int yCount = 0;
	int move = 4;
	int dim = 5;
	int numImages = 400;
	float *** data;
	data = new float **[numImages];
	for(int i = 0; i<numImages; i++){
		data[i]= new float*[644];
	}
	for (int i = 0; i<numImages; i++){
		for(int j = 0; j<644; j++){
			data[i][j] = new float[network.weight_dim];
		}
	}
	//output = new BMP[644];

	int pixelLocation= 0;
	int xCount = 0;
	int pictureLoopCounter = 1;
	int subjectCounter =1;
	for(int i = 1; i<=numImages; i++){
		string s = int_to_str(pictureLoopCounter);
		string s1 = int_to_str(subjectCounter);
		if(pictureLoopCounter%10 != 0){
			pictureLoopCounter++;
		}else{
			pictureLoopCounter = 1;
			subjectCounter++;
		}

		string final = "att_faces/s"+s1+"/"+s+".bmp";
		//cout <<final << endl;
		input.ReadFromFile(final.c_str());
		while (pos.endY < imageY){


			if (yCount  ==0) pos.startY = 0;
			else pos.startY = (move)*(yCount-1)+(dim-1);
			pos.endY = yCount*(move)+dim;
			//	cout << pos.startY << "   " << pos.endY << endl;
			if(pos.endY > imageY){
				pos.endY = imageY;
				pos.startY = pos.endY-dim;
			}
			while(pos.endX < imageX){

				if (xCount ==0) pos.startX = 0;
				else pos.startX = (move)*(xCount-1)+(dim-1);
				pos.endX = xCount*(move)+dim;
				if(pos.endX > imageX){
					pos.endX = imageX;
					pos.startX = pos.endX-dim;
				}

				xCount++;
				//	cout << pos.startX << "   " << pos.endX<<"----"<<pos.startY << "   " << pos.endY << endl;

				for (int k = pos.startY; k<pos.endY; k++){
					for (int l = pos.startX; l<pos.endX; l++){

						data[i-1][numIterationsCompletedCount][pixelLocation] = (float)input(l, k)->Red;
						pixelLocation++;
					}

				}
				numIterationsCompletedCount++;
				pixelLocation = 0;
			}

			pos.startX = 0;
			pos.endX = 0;
			xCount = 0;
			yCount++;

		}
		xCount = 0;
		yCount = 0;

		pos.startX = 0;
		pos.startY = 0;
		pos.endX = 0;
		pos.endY = 0;
		//cout << numIterationsCompletedCount << endl;
		numIterationsCompletedCount = 0;
		pixelLocation = 0;
	}  
	for(int imageNum = 0; imageNum < 10; imageNum++){
		for(int i = 0; i<644; i++){
			float distance = 0.0f;
			for (int j = 0; j<network.weight_dim; j++){

				distance += abs(data[imageNum][i][j])*abs(data[imageNum][i][j]);
			}
			distance = sqrt(distance);
			for (int j = 0; j<network.weight_dim; j++){
				if (distance !=0){
					data[imageNum][i][j] /= distance;
				}else{
					data[imageNum][i][j] = 0;
				}

			}
		}
	}

	cout << "done"<<  "   "<< yCount << endl;

	//cout << data[0][24] << endl;
	network.train(data);
	cout << "done with first stage" << endl;
	network.train(data, 2, .02, 500000);
	BMP **final;
	final = new BMP*[numImages];
	for(int i = 0; i< numImages; i++){
		final[i] = new BMP[3];
	}
	for(int i = 0; i<numImages; i++){
		for(int j = 0; j<3; j++){
			final[i][j].SetSize(23, 28);
		}
	}
	int count  = 0;
	pictureLoopCounter = 1;
	subjectCounter = 1;
	neuron temp;
	RGBApixel pixel;
	
	for(int k = 0; k<numImages; k++){
		//for(int m = 0;m<3; m++){
		for (int i = 0;i<28; i++){
			for(int j = 0; j< 23; j++){


				temp = network.findBMU(data[k][count]);

				pixel.Red = temp.X*(255/network.mapHeight);
				pixel.Blue = temp.X*(255/network.mapHeight);
				pixel.Green = temp.X*(255/network.mapHeight);
				pixel.Alpha = 0;
				final[k][0].SetPixel(j,i, pixel);
				pixel.Red = temp.Y*(255/network.mapHeight);
				pixel.Blue = temp.Y*(255/network.mapHeight);
				pixel.Green = temp.Y*(255/network.mapHeight);
				pixel.Alpha = 0;
				final[k][1].SetPixel(j, i, pixel);
				pixel.Red = temp.Z*(255/network.mapHeight);
				pixel.Blue = temp.Z*(255/network.mapHeight);
				pixel.Green = temp.Z*(255/network.mapHeight);
				pixel.Alpha = 0;
				final[k][2].SetPixel(j, i, pixel);

				count++;

			}
		}
		count = 0;

		//string s1 = int_to_str(0);
		
		string s = int_to_str(pictureLoopCounter);
		string s2 = int_to_str(subjectCounter);
		string filename = "mkdir C:\\C++\\SOM\\SOM\\images\\s"+s2+"\\"+s+"\\";

		wstring stemp = wstring(filename.begin(), filename.end());
		LPCWSTR final_string = stemp.c_str();
		system(filename.c_str());
		//	CreateDirectory(final_string, NULL);

		string filename1 = "images/s"+s2+"/"+s+"/0.bmp";
		final[k][0].WriteToFile(filename1.c_str());
		filename1 = "images/s"+s2+"/"+s+"/1.bmp";
		final[k][1].WriteToFile(filename1.c_str());
		filename1 = "images/s"+s2+"/"+s+"/2.bmp";
		final[k][2].WriteToFile(filename1.c_str());
		
			if(pictureLoopCounter%10!=0){
				pictureLoopCounter++;
			}else{
				pictureLoopCounter = 1;
				subjectCounter++;

			}
		
		//}




	}
	return 0;
}