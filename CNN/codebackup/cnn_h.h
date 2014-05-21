for (int i = 0; i<layers[numLayers-1].numNeurons; i++){


					layers[numLayers-1].neurons[i].errorFactor = layers[numLayers-1].neurons[i].output-layers[numLayers-1].neurons[i].expected;


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

							//cout << "hi" << endl;
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
							//cout << "hi" << endl;


							int weight;
							int previousLayerID;
							//int i = rand()%layers[n].numNeurons;
							for(int i = 0; i<layers[n].numNeurons; i++){
								
								for(unsigned int j = 0; j<layers[n].neurons[i].inputWeightID.size(); j++){

									weight = layers[n].neurons[i].inputWeightID[j];
									size = layers[n].mapSize;
									previousLayerID = layers[n].neurons[i].inputs[j];

									layers[n].data[layers[n].neurons[i].featureMapID].weights[weight/size][weight%size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[i].delta*1;

								}
							}

						}else if(layers[n+1].getType() == "s"&&layers[n+2].getType() == "c"){

							//cout << "hi" << endl;
							for(int i = 0; i<layers[n].numNeurons; i++){
								int currentLayerID = i;
								int nextLayerID;
								int weight;
								//cout << n << endl;
								for(unsigned int j = 0; j<layers[n].neurons[currentLayerID].outputs.size(); j++){
									for(unsigned int k = 0; k<layers[n+1].neurons[layers[n].neurons[currentLayerID].outputs[j]].outputs.size(); k++){
										// since there are no weights in a subsampling layer, i am looking at the weights in the next layer to calculate the delta
										nextLayerID = layers[n+1].neurons[layers[n].neurons[currentLayerID].outputs[j]].outputs[k];
										weight = layers[n+1].neurons[layers[n].neurons[currentLayerID].outputs[j]].outputWeightID[k];
										size = layers[n+2].mapSize;

										layers[n].neurons[currentLayerID].errorFactor += layers[n+2].neurons[nextLayerID].delta*
											layers[n+2].data[layers[n+2].neurons[nextLayerID].featureMapID].weights[weight/size][weight%size];
										if(n==1) {
											//cout << "hi\n";
										}

									}
								}
							}
							for(int i = 0; i<layers[n].numNeurons; i++){
								layers[n].neurons[i].delta = layers[n].neurons[i].output * (1-layers[n].neurons[i].output) * layers[n].neurons[i].errorFactor *1;
								layers[n].neurons[i].bias += learningRate*layers[n].neurons[i].delta*1;
							}
							//cout << "hi" << endl;


							int weight;
							int previousLayerID;
							for(int i = 0; i<layers[n].numNeurons; i++){
							//int i = rand()%layers[n].numNeurons;
								for(unsigned int j = 0; j<layers[n].neurons[i].inputWeightID.size(); j++){

									weight = layers[n].neurons[i].inputWeightID[j];
									size = layers[n].mapSize;
									previousLayerID = layers[n].neurons[i].inputs[j];

									layers[n].data[layers[n].neurons[i].featureMapID].weights[weight/size][weight%size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[i].delta *1;

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
										layers[n+1].data[layers[n+1].neurons[nextLayerID].featureMapID].weights[weight/size][weight%size];

								}
							}
							int currentNeuronID = rand()%layers[n].numNeurons;

							layers[n].neurons[currentNeuronID].delta = layers[n].neurons[currentNeuronID].output * (1-layers[n].neurons[currentNeuronID].output) * layers[n].neurons[currentNeuronID].errorFactor *1;
							layers[n].neurons[currentNeuronID].bias += learningRate*layers[n].neurons[currentNeuronID].delta*1;




							int weight;
							int previousLayerID;
							for(unsigned int j = 0; j<layers[n].neurons[currentNeuronID].inputWeightID.size(); j++){
								weight = layers[n].neurons[currentNeuronID].inputWeightID[j];
								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[currentNeuronID].inputs[j];
								layers[n].data[layers[n].neurons[currentNeuronID].featureMapID].weights[weight/size][weight%size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[currentNeuronID].delta *1;
								
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
							int currentNeuronID = rand()%layers[n].numNeurons;
							layers[n].neurons[currentNeuronID].delta = layers[n].neurons[currentNeuronID].output * (1-layers[n].neurons[currentNeuronID].output) * layers[n].neurons[currentNeuronID].errorFactor *1;
							layers[n].neurons[currentNeuronID].bias += learningRate*layers[n].neurons[currentNeuronID].delta*1;




							int weight;
							int previousLayerID;
							for(unsigned int j = 0; j<layers[n].neurons[currentNeuronID].inputWeightID.size(); j++){
								weight = layers[n].neurons[currentNeuronID].inputWeightID[j];
								size = layers[n].mapSize;
								previousLayerID = layers[n].neurons[currentNeuronID].inputs[j];

								layers[n].data[layers[n].neurons[currentNeuronID].featureMapID].weights[weight/size][weight%size] += learningRate*layers[n-1].neurons[previousLayerID].output * layers[n].neurons[currentNeuronID].delta *1;


							}


						}
					}
				}
						//reset();
				propagate();
				learningRate += -learningRate/((double)numIterations);

				layers[numLayers-1].neurons[input_data[j].subject_num-1].expected = 0.0f;
				cout << j << endl;

				
			}
		}
	}