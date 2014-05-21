#ifndef CNN_H
#define CNN_H
using namespace std;
class Neuron{
public:
	vector<int> inputs;
	vector<int> inputWeightID;
	vector<int> outputs;
	vector<int> outputWeightID;
	double expected;
	double bias; //used to propagate
	double input; // input of the neuron
	double errorFactor; //error factor, used in backprop
	double delta; //used in backprop
	int numInputs; //The number of inputs to the neuron, used in for loops
	double output; // output of the neuron
	int numOutputs;
	int featureMapID;
	Neuron(){
		output = 0.0f;
		numInputs = 0;
		numOutputs = 0;
		delta = 0.0f;
		int flag = rand()%100;
		if(flag  %2 == 0){
			bias = ((double)rand()/RAND_MAX)*.1;
		}else{
			bias = ((double)rand()/RAND_MAX)*(-.1);
		}
		errorFactor = 0.0f;
		input = 0.0f;
		expected = 0.0f;
		featureMapID = 0;
	}
	void giveInputConnection(int connect){
		inputs.push_back(connect);
	}
	void setMapID(int ID){
		featureMapID = ID;
	}

};

class Weights{
public:
	double ** weights;
	string type;
	Weights(int mapX, int mapY, string type1){

		type = type1;
		weights = new double*[mapX];
		int flag = 0;
		for(int i = 0; i<mapX; i++){
			weights[i] = new double[mapY];

		}

		if(type == "c"){
			for(int i = 0; i<mapX; i++){
				for(int j = 0; j<mapY; j++){
					flag = rand()%100;
					if(flag%2==0){
						weights[i][j] = ((double)rand()/RAND_MAX)*.4;
					}else{
						weights[i][j] = (-(double)rand()/RAND_MAX)*.4;
					}
				}
			}
		}else{
			for(int i = 0; i<mapX; i++){
				for(int j = 0; j<mapY; j++){

					flag = rand()%100;
					if(flag%2==0){
						weights[i][j] = ((double)rand()/RAND_MAX)*.4;
					}else{
						weights[i][j] = (-(double)rand()/RAND_MAX)*.4;
					}
				}
			}
		}

	}
	Weights(){
	}
};
class NeuronLayer{
private:
	string type;
public:
	int mapSize;
	Neuron * neurons;
	int numOfMaps;
	int move;
	Weights *data;
	int numNeurons;
	int x_count;
	int y_count;
	NeuronLayer(string t,int numberOfFeatureMaps, int featureMapSize,int numNeuronPerFeatureMap, int move1){
		if(numberOfFeatureMaps !=0) {
			numNeurons = numNeuronPerFeatureMap*numberOfFeatureMaps;
		}else{
			numNeurons = numNeuronPerFeatureMap;
		}
		mapSize = featureMapSize;
		move = move1;
		numOfMaps = numberOfFeatureMaps;
		type = t;
		if(type=="c" || type == "s"){
			data = new Weights[numOfMaps]; 
		}else{
			data = new Weights[1];
		}
		x_count = 0;
		y_count =0;
		neurons = new Neuron[numNeurons];

	}
	NeuronLayer(){
	}
	void setType(string t){
		type = t;
	}
	string getType(){
		return type;
	}





};

class CNN{
public:
	int numLayers;
	ifstream config;
	NeuronLayer *layers;
	int **data;
	struct image_data{
		int subject_num;
		int face_position;
		int face_num;
		float ** data;
	};
	image_data * input_data;
	int numHidden;
	string int_to_str(int i){
		stringstream ss;
		ss << i;
		string str = ss.str();
		return str;
	}

	double sigmoid(double x){
		return tanh(x);
	}

	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
		return elems;
	}


	std::vector<std::string> split(const std::string &s,  char delim) {
		std::vector<std::string> elems;
		split(s, delim, elems);
		return elems;
	}


	CNN(char * filename, int numLayer){



		numLayers = numLayer;
		numHidden = numLayers-2;
		string s;
		vector<string> buffer;
		int count = 0;
		layers = new NeuronLayer[numLayer];
		config.open(filename, ios::in);
		char delim = ' ';
		cout << "Reading config file...";
		if(config.is_open()){
			while(getline(config, s)){

				// type numFeatureMaps featureMapSize numNeurons

				if(s.c_str()[0] != '#'){
					buffer = split(s, delim);

					layers[count] = *(new NeuronLayer(buffer[0], atoi(buffer[1].c_str()), atoi(buffer[2].c_str()), atoi(buffer[3].c_str()), atoi(buffer[4].c_str())));
					count++;
				}
			}
		}
		cout << "Done \n";
		cout << "Initializing Network...";
		buffer.empty();
		//layers[1].initConnections(23, 28);

		/*for(int j = 0; j<layers[1].numOfMaps; j++){
		layers[1].data[j] = *(new Weights(layers[1].mapSize, layers[1].mapSize,layers[1].getType()));

		}*/
		struct boxPos{
			int startX;
			int startY;
			int endX;
			int endY;
		};
		layers[0].x_count = 23;
		layers[0].y_count = 28;
		for(int i = 1; i<numLayers; i++){

			if(layers[i].getType() == "c" || layers[i].getType() == "s"){


				int numFeatureMapsInPreviousLayer = layers[i-1].numOfMaps;
				int numNeuronInPreviousLayer = layers[i-1].numNeurons/layers[i-1].numOfMaps;

				boxPos pos;
				int greatest = 0;
				layers[i].x_count = 0;
				layers[i].y_count = 0;
				pos.endX = 0;
				pos.endY = 0;
				pos.startX = 0;
				pos.startY = 0;
				int featureMapCount = 0;
				int move = layers[i].move;

				int count = 0;
				int yCount = 0;
				int xCount = 0;
				int numIterationsCompletedCount = 0;

				int imageY = layers[i-1].y_count;
				int imageX = layers[i-1].x_count;
				int actualPos;
				int pixelLocation = 0;
				float ** data;
				for(int set = 0; set<=(numFeatureMapsInPreviousLayer-1)*numNeuronInPreviousLayer; set+=numNeuronInPreviousLayer){
					while (pos.endY < imageY){

						if(set == 0) layers[i].x_count = 0;
						if(set == 0) layers[i].y_count++;
						pos.startY = (move)*(yCount);
						pos.endY = yCount*(move)+layers[i].mapSize;


						if(pos.endY > imageY){
							pos.endY = imageY;
							pos.startY = pos.endY-layers[i].mapSize;
						}
						while(pos.endX < imageX){

							if(set == 0) layers[i].x_count++;
							pos.startX = (move)*(xCount);
							pos.endX = xCount*(move)+layers[i].mapSize;

							if(pos.endX > imageX){
								pos.endX = imageX;
								pos.startX = pos.endX-layers[i].mapSize;
							}

							xCount++;
							for (int k = pos.startY; k<pos.endY; k++){
								for (int l = pos.startX; l<pos.endX; l++){
									actualPos = (k)*imageX+l+set;

									for (int m = 1; m< (layers[i].numNeurons); m+=(layers[i].numNeurons/layers[i].numOfMaps)){
										if(numIterationsCompletedCount+m-1>0){

											layers[i].neurons[numIterationsCompletedCount+m-1].setMapID(featureMapCount);
											layers[i].neurons[numIterationsCompletedCount+m-1].giveInputConnection(actualPos);

											layers[i-1].neurons[actualPos].outputWeightID.push_back(pixelLocation);
											layers[i-1].neurons[actualPos].outputs.push_back(numIterationsCompletedCount+m-1);
											layers[i].neurons[numIterationsCompletedCount+m-1].inputWeightID.push_back(pixelLocation);
											count++;
										}else{
											layers[i-1].neurons[actualPos].outputWeightID.push_back(pixelLocation);
											layers[i].neurons[0].setMapID(featureMapCount);
											layers[i].neurons[0].giveInputConnection(actualPos);

											layers[i].neurons[0].inputWeightID.push_back(pixelLocation);
											layers[i-1].neurons[actualPos].outputs.push_back(0);
											count++;
										}
										featureMapCount++;
										//cout << pixelLocation << endl;

										pixelLocation++;
										pixelLocation %= layers[i].mapSize*layers[i].mapSize;


									}

									featureMapCount=0;
									count = 0;

								}

							}	
							numIterationsCompletedCount++;
							//pixelLocation = 0;

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
					numIterationsCompletedCount = 0;
					pixelLocation = 0;

				}
				for(int j = 0; j<layers[i].numOfMaps; j++){
					if(layers[i].getType() != "s"){
						layers[i].data[j] = *(new Weights(layers[i].mapSize, layers[i].mapSize,layers[i].getType()));
					}
				}
			}
			if(layers[i].getType() == "r"){
				layers[i].data[0] = *(new Weights(layers[i].numNeurons, layers[i-1].numNeurons, layers[i].getType()));
			}
		}

		cout << "Done \n";

	}
	void reset(){
		for(int i = 1; i<numLayers; i++){
			for(int j = 0; j<layers[i].numNeurons; j++){
				layers[i].neurons[j].output = 0.0;
			}
		}
	}
	void propagateSigmoid(){
		reset();
		int count = 0;
		int xpos;
		int ypos;
		int connectionsID;
		int mapID;
		for (int i = 1; i<numLayers; i++){
			if(layers[i].getType() == "c" || layers[i].getType() == "s"){

				for (int j = 0; j<layers[i].numNeurons; j++){
					//if(i==2)cout << layers[i].neurons[j].inputs.size() << "\n";
					for(unsigned int k = 0; k<layers[i].neurons[j].inputs.size(); k++){
						if(layers[i].getType() == "c"){

							connectionsID = layers[i].neurons[j].inputs[k];
							mapID = layers[i].neurons[j].featureMapID;
							xpos = count%layers[i].mapSize;
							ypos = count/layers[i].mapSize;
							layers[i].neurons[j].output += layers[i-1].neurons[connectionsID].output*
								layers[i].data[mapID].weights[xpos][ypos];
							count++;
							count %= layers[i].mapSize*layers[i].mapSize;
						}else{
							connectionsID = layers[i].neurons[j].inputs[k];

							layers[i].neurons[j].output += layers[i-1].neurons[connectionsID].output*.01;

						}
					}
					count = 0;
					if(layers[i].getType() == "s"){
						//layers[i].neurons[j].output /= layers[i-1].mapSize*layers[i-1].mapSize;
						//layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
					}else{
						layers[i].neurons[j].output += layers[i].neurons[j].bias;
						layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
					}

				}

			}else if (layers[i].getType() == "r"){
				for (int j = 0; j<layers[i].numNeurons; j++){
					for(int k = 0; k<layers[i-1].numNeurons; k++){

						layers[i].neurons[j].output += layers[i].data[0].weights[j][k]*layers[i-1].neurons[k].output;
					}
					layers[i].neurons[j].output += layers[i].neurons[j].bias;
					layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
				}
			}
		}
	}
	void propagate(){
		reset();
		int count = 0;
		int xpos;
		int ypos;
		int connectionsID;
		int mapID;
		for (int i = 1; i<numLayers; i++){
			if(layers[i].getType() == "c" || layers[i].getType() == "s"){

				for (int j = 0; j<layers[i].numNeurons; j++){
					//if(i==2)cout << layers[i].neurons[j].inputs.size() << "\n";
					for(unsigned int k = 0; k<layers[i].neurons[j].inputs.size(); k++){
						if(layers[i].getType() == "c"){

							connectionsID = layers[i].neurons[j].inputs[k];
							mapID = layers[i].neurons[j].featureMapID;
							xpos = count%layers[i].mapSize;
							ypos = count/layers[i].mapSize;
							layers[i].neurons[j].output += layers[i-1].neurons[connectionsID].output*
								layers[i].data[mapID].weights[xpos][ypos];
							count++;
							count %= layers[i].mapSize*layers[i].mapSize;
						}else{
							connectionsID = layers[i].neurons[j].inputs[k];
							
							layers[i].neurons[j].output += layers[i-1].neurons[connectionsID].output;
							//cout << layers[i-1].neurons[connectionsID].output << endl;

						}
					}
					count = 0;
					if(layers[i].getType() == "s"){
						//cout << "hi\n";
						layers[i].neurons[j].output /= layers[i].neurons[j].inputs.size();
						//layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
					}else{
						layers[i].neurons[j].output += layers[i].neurons[j].bias;
						layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
					}

				}

			}else if (layers[i].getType() == "r"){
				for (int j = 0; j<layers[i].numNeurons; j++){
					for(int k = 0; k<layers[i-1].numNeurons; k++){

						layers[i].neurons[j].output += layers[i].data[0].weights[j][k]*layers[i-1].neurons[k].output;
					}
					layers[i].neurons[j].output += layers[i].neurons[j].bias;
					layers[i].neurons[j].output = sigmoid(layers[i].neurons[j].output);
				}
			}
		}
	}

	void backprop(float expected){

		float learningRate = -.009f;
		int size = 0;

		for(int z = 1; z<5; z++){

			for (int i = 0; i<layers[numLayers-1].numNeurons; i++){


				layers[numLayers-1].neurons[i].errorFactor = layers[numLayers-1].neurons[i].output-layers[numLayers-1].neurons[i].expected;

				//cout << layers[numLayers-1].neurons[i].output << endl;
				layers[numLayers-1].neurons[i].delta = layers[numLayers-1].neurons[i].output * (1-	layers[numLayers-1].neurons[i].output) * (layers[numLayers-1].neurons[i].output*layers[numLayers-1].neurons[i].errorFactor) ;
				layers[numLayers-1].neurons[i].bias += learningRate*layers[numLayers-1].neurons[i].delta * 1;
			}

			for(int m = 0; m<layers[numLayers-1].numNeurons; m++){
				for(int k = 0; k<layers[numLayers-2].numNeurons; k++){
					layers[numLayers-1].data[0].weights[m][k] += learningRate * layers[numLayers-2].neurons[k].output * layers[numLayers-1].neurons[m].delta * 1;

				}
			}

			for (int n = numHidden; n>0; n--){
				if(layers[n].getType() == "r"){
					for(int i = 0; i<layers[n].numNeurons; i++){
						for(int j = 0; j<layers[n+1].numNeurons; j++){
							layers[n].neurons[i].errorFactor += layers[n+1].neurons[j].delta * layers[n+1].data[0].weights[j][i];

						}
					}
					for(int i = 0; i<layers[n].numNeurons; i++){
						layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
						layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
					}

					for(int m = 0; m<layers[n].numNeurons; m++){
						for(int k = 0; k<layers[n-1].numNeurons; k++){

							layers[n].data[0].weights[m][k] += learningRate * layers[n-1].neurons[k].output * layers[n].neurons[m].delta *1;

						}
					}
				}else if (layers[n].getType() == "c"){
					if(layers[n+1].getType() == "s"&&layers[n+2].getType() == "r"){


						for(int i = 0; i<layers[n].numNeurons; i++){

							for(unsigned int j = 0; j<layers[n].neurons[i].outputs.size(); j++){
								for(unsigned int k = 0; k<layers[n+2].numNeurons; k++){
									// since there are no weights in a subsampling layer, i am looking at the weights in the next layer to calculate the delta


									layers[n].neurons[i].errorFactor += layers[n+2].neurons[k].delta*
										layers[n+2].data[0].weights[k][layers[n].neurons[i].outputs[j]];


								}
							}
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
							layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
						}


						int weight = 0;
						int previousLayerID;
						
						for(int i = 0; i<layers[n].numNeurons; i++){

							for(unsigned int j = 0; j<layers[n].neurons[i].inputs.size(); j++){

								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[i].inputs[j];

								layers[n].data[layers[n].neurons[i].featureMapID].weights[weight%size][weight/size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[i].delta*1;
								weight++;
								weight %= layers[n].mapSize*layers[n].mapSize;
							}
						}
					}else if(layers[n+1].getType() == "s"&&layers[n+2].getType() == "c"){
						//cout << "hi" << endl;
						for(int i = 0; i<layers[n].numNeurons; i++){
							int currentLayerID = i;
							int nextLayerID;
							int weight = 0;
							for(unsigned int j = 0; j<layers[n].neurons[currentLayerID].outputs.size(); j++){
								for(unsigned int k = 0; k<layers[n+1].neurons[layers[n].neurons[currentLayerID].outputs[j]].outputs.size(); k++){
									nextLayerID = layers[n+1].neurons[layers[n].neurons[currentLayerID].outputs[j]].outputs[k];
									
									size = layers[n+2].mapSize;
									
									layers[n].neurons[currentLayerID].errorFactor += layers[n+2].neurons[nextLayerID].delta*
										layers[n+2].data[layers[n+2].neurons[nextLayerID].featureMapID].weights[weight%size][weight/size];
									if(n==3) {
										cout << "hi\n";
									}
									weight++;
									weight%=layers[n+2].mapSize*layers[n+2].mapSize;

								}
							}
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
							layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
						}

						for(int i = 0; i<layers[n].numNeurons; i++){
							int currentNeuronID = i;
							int weight = 0;
							int previousLayerID;

							for(unsigned int j = 0; j<layers[n].neurons[currentNeuronID].inputWeightID.size(); j++){

								
								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[currentNeuronID].inputs[j];

								layers[n].data[layers[n].neurons[currentNeuronID].featureMapID].weights[weight%size][weight/size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[currentNeuronID].delta *1;
								weight++;
								weight %= layers[n].mapSize*layers[n].mapSize;
							}
						}

					}else if(layers[n+1].getType() == "c"){

						//++++++++++++++++++++++++++++++++++
						//this is the case where you are on a convolutional layer, surrounded by 2 convolution layers. I first 
						//randonmly select a neuron. to calculate delta, first i use the "outputs" vector to find which neurons are connected 
						//to this neuron in the next layer. I multiply the deltas of those neurons by the output of those neurons, and 
						//get the error factor. I then use the standard formula for calculating delta. I then use the inputs vector to find which weights
						//this neuron is using, and adjust those accordingly, by the rule of gradient descent.
						//++++++++++++++++++++++++++++++++++
						for(int i = 0; i<layers[n].numNeurons; i++){
							int currentLayerID = i;
							int nextLayerID;
							int weight;
							for(unsigned int j = 0; j<layers[n].neurons[currentLayerID].outputs.size(); j++){
								nextLayerID = layers[n].neurons[currentLayerID].outputs[j];
								weight = layers[n].neurons[currentLayerID].outputWeightID[j];
								size = layers[n+1].mapSize;

								layers[n].neurons[currentLayerID].errorFactor += layers[n+1].neurons[nextLayerID].delta*
									layers[n+1].data[layers[n+1].neurons[nextLayerID].featureMapID].weights[weight%size][weight/size];

							}
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
							layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							int currentNeuronID = i;
							int weight;
							int previousLayerID;
							for(unsigned int j = 0; j<layers[n].neurons[currentNeuronID].inputWeightID.size(); j++){
								weight = layers[n].neurons[currentNeuronID].inputWeightID[j];
								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[currentNeuronID].inputs[j];
								layers[n].data[layers[n].neurons[currentNeuronID].featureMapID].weights[weight%size][weight/size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[currentNeuronID].delta *1;

							}
						}

						//++++++++++++++++++++++++++++++++++
						//THIS IS WHERE I LEFT OFF - i am trying to program the case where you are on a convolutional layer, but the layer in front of you is fully connected
						//and the previous layer is convolutional. So, you use the deltas of the fully connected layer to calculate the error factor of each neuron in the current
						//convolutional layer. Once I have done that, I can select neurons in the current convolutional layer RANDOMLY and adjust the weight attached to that neuron 
						//accordingly
						//++++++++++++++++++++++++++++++++++
					}else if(layers[n+1].getType() == "r"){
						for(int i = 0; i<layers[n].numNeurons; i++){
							for(int j = 0; j<layers[n+1].numNeurons; j++){
								layers[n].neurons[i].errorFactor += layers[n+1].neurons[j].delta * layers[n+1].data[0].weights[j][i];



							}
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
							layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
						}
						for(int i = 0; i<layers[n].numNeurons; i++){
							int currentNeuronID = i;
							int weight;
							int previousLayerID;
							for(unsigned int j = 0; j<layers[n].neurons[currentNeuronID].inputWeightID.size(); j++){
								weight = layers[n].neurons[currentNeuronID].inputWeightID[j];
								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[currentNeuronID].inputs[j];

								layers[n].data[layers[n].neurons[currentNeuronID].featureMapID].weights[weight%size][weight/size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[currentNeuronID].delta *1;


							}
						}

					}
				}

			}

			propagate();
			learningRate += -learningRate/((double)20);
			//cout << learningRate << endl;
		}



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
					input_data[count].data = new float*[23];
					for(int l = 0; l<23; l++){
						input_data[count].data[l] = new float[28];
					}
					for(int l = 0; l< 23; l++){
						for(int m = 0; m<28; m++){
							input_data[count].data[l][m] = (((float)image(l, m)->Red)/255);
						}
					}
					count++;
				}
			}
		}

	}

	void loadImage(int num){
		int count = 0;
		for(int k = 0; k<23; k++)
		{
			for(int l = 0; l<28; l++)
			{
				layers[0].neurons[count].output = input_data[num].data[k][l];
				count++;
			}
		}
		for(int k = 0; k<23; k++)
		{
			for(int l = 0; l<28; l++)
			{
				layers[0].neurons[count].output = input_data[num+1].data[k][l];
				count++;
			}
		}
		for(int k = 0; k<23; k++)
		{
			for(int l = 0; l<28; l++)
			{

				layers[0].neurons[count].output = input_data[num+2].data[k][l];
				count++;
			}
		}
	}


	/*
	void train(int num_iterations){
	for(int i = 0; i<num_iterations; i++){
	for(int j = 0; j<40; j++){

	*/

};
#endif