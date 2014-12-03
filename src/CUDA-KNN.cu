//
//  CUDA-KNN.cu
//
//  Created by Ishbir Singh on 22/06/12.
//  Copyright (c) 2012-2014 webmaster@ishbir.com. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <chrono>

#include "Definitions.cuh"
#include "kernels/segmentedsort.cuh"

using namespace mgpu;
using namespace std;

__global__ void populate_keys_with_training_ids(int* keys) {
	/**
	* Populates a keys array with numbers from 0 to train_count
	*/
	int global_id_0 = blockIdx.x * blockDim.x + threadIdx.x; // ID of test point
	int global_id_1 = blockIdx.y * blockDim.y + threadIdx.y; // ID of training point
	int global_size_0 = gridDim.x * blockDim.x; // Number of test points
	int global_size_1 = gridDim.y * blockDim.y; // Number of training points

	int id = global_id_0*global_size_1 + global_id_1;
	keys[id] = global_id_1;
}

__global__ void populate_segments(int* segments, int train_count) {
	/**
	* Populates a segments array with indices of where each new test point begins (segments in sorting)
	*/
	int global_id_0 = blockIdx.x * blockDim.x + threadIdx.x; // ID of test point
	segments[global_id_0] = global_id_0*train_count;
}

__global__ void distances_computation(float* test_g, float* train_g, float *output, int dims) { // dims is dimensions of data
	/**
	* test_g: Array of test points in global memory
	* train_g: Array of training points in global memory
	* output: Array of output distance calculations in global memory
	* dims: Number of dimensions in the incoming data
	*/

	float res = 0; // Stores the final result
	int global_id_0 = blockIdx.x * blockDim.x + threadIdx.x; // ID of test point
	int global_id_1 = blockIdx.y * blockDim.y + threadIdx.y; // ID of training point
	int global_size_0 = gridDim.x * blockDim.x; // Number of test points
	int global_size_1 = gridDim.y * blockDim.y; // Number of training points

	extern __shared__ float test[];
	if (threadIdx.y < dims) { // first 'dims' threads copy each dimension float to local memory
		test[threadIdx.y] = test_g[dims*global_id_0 + threadIdx.y];
	}
	__syncthreads(); // wait for copy operation

	for (int i=0; i < dims; i++) { // loop over!
		res += pow((train_g[global_size_1*i+global_id_1] - test[i]), 2); // find the right train point to use
	}

	int id = global_id_0*global_size_1 + global_id_1; // ID of test point*Number of training points + Training point ID

	// Thus, the corresponding distances between one test point and all training points are stored in a contiguous location
	// This approach is very useful for segmented sorting.

	output[id] = res;
}

///
/// This is the entry point of the application
/// Accepted filetypes: CSV
///
int main(int argc, char* argv[]) {
	if (argc != 4) // File name is not present or malformed arguments,
	{
		cerr << "Invalid arguments. Arguments are: test_data_file training_data_file num_types_train" << endl;
		return 1;
	}

	/**
	Read the test data from specified file

	Test Data Format:
	-----------------

	x1,x2,x3,x4
	x1,x2,x3,x4
	x1,x2,x3,x4
	x1,x2,x3,x4
	y1,y2,y3,y4
	y1,y2,y3,y4
	y1,y2,y3,y4
	z1,z2,z3,z4
	z1,z2,z3,z4
	**/
	int test_count, col_count;

	ifstream infile(argv[1]); // Open file
	test_count = count(istreambuf_iterator<char>(infile), istreambuf_iterator<char>(), '\n')+1; // count number of points
	infile.seekg(0); // seek back

	string line; // temp var for one line
	getline(infile, line); // read one line

	stringstream firstStream(line, stringstream::in | stringstream::out); // make a stream
	col_count = count(istreambuf_iterator<char>(firstStream), istreambuf_iterator<char>(), ',')+1; // count number of dimensions [2 commas mean 3 dims]

	float* test_points = new float[test_count*col_count];

	for(unsigned int i=0; i < test_count/* && !infile.eof() && infile.good()*/; i++) { // Keep reading and storing
		stringstream lineStream(line, stringstream::in | stringstream::out); // make a stream

		string cell;
		float val;

		for(unsigned int j=0; j < col_count; j++) { // we don't want to store the last thing which gives class
			getline(lineStream, cell, ',');
			if (cell == "") // empty
				continue;
			from_string<float>(val, cell, std::dec);
			test_points[i*col_count+j] = val; // Convert to float and store
		}
		getline(infile, line); // read one line
	}

	// Cleanup
	infile.close(); // We are done

	/** Done with reading test data; Start reading training data **/
	/**
	Train Data Format:
	-----------------

	x1,x2,x3,x4
	x1,x2,x3,x4
	x1,x2,x3,x4
	x1,x2,x3,x4

	y1,y2,y3,y4
	y1,y2,y3,y4
	y1,y2,y3,y4

	z1,z2,z3,z4
	z1,z2,z3,z4

	This tells us that the test data has 3 classifications.

	Train Data Array Format:
	------------------------------------

	[x1,x2,x3,x4, ... y1,y2,y3,y4, ... z1,z2,z3,z4]

	Train Data Classification Vector Format: [k is index]
	---------------------------------------------------

	( index: [classifcation_start_index, classification_end_index] )

	k0: [0, 19],
	k1: [20, 39],
	k2: [40, 59]

	k0, k1 and k2 give the various classes of the test data to compare to.

	Both these arrays have been kept separate to reduce complexity of code and to maintain a 2 x 2 matrix in both test and train set as specified in paper.
	**/

	ifstream trainfile(argv[2]);

	int train_count, train_line_count;

	// Count the lines and the types
	train_line_count = count(istreambuf_iterator<char>(trainfile), istreambuf_iterator<char>(), '\n')+1; // count last line also
	trainfile.seekg(0);
	trainfile.clear(); // clear EOF bit

	int num_types; // num of types of training points
	from_string<int>(num_types, (string)argv[3], std::dec);

	train_count = train_line_count-num_types+1; // +1 is because of the fact that if there are 2 types, there will be only 1 "\n\n"

	float* train_points = new float[train_count*col_count];

	vector<vector<int>> train_points_classes(num_types, vector<int>(2, 0)); // Init vector for keeping track of classes

	unsigned int type_count = 0; // Keep track of the type id

	train_points_classes[0][0] = 0; // Starting point is 0

	for (int i=0; i < train_count /*&& !trainfile.eof() && trainfile.good()*/; i++) { // Keep reading and storing
		string line;
		getline(trainfile, line);

		if (line == "" && type_count < num_types) { // classification boundary
			train_points_classes[type_count][1] = i-1;

			if (type_count == num_types-1) // last type so set its ending beforehand
				train_points_classes[type_count][1] = train_count*col_count - 1; // common sense
			else
				train_points_classes[type_count][0] = i; // set beginning of next type
			type_count += 1; // increment
			i--; // compensation necessary
		}
		else { 
			stringstream lineStream(line, stringstream::in | stringstream::out); // make a stream

			string cell;
			float val;

			for(int j=0; j < col_count; j++) {
				getline(lineStream, cell, ',');
				if (cell == "") // empty
					continue;
				from_string<float>(val, cell, std::dec);
				train_points[j*train_count + i] = val; // Convert to float and store
			}
		}
	}
	trainfile.close();

	/** Done reading all the data **/
	cout << "Test Count: " << test_count<< "\n";
	cout << "Train Count: " << train_count << "\n";
	cout << "Dimensions: " << col_count << "\n\n";

	// Main stuff comes here
	ContextPtr context = CreateCudaDevice(0);

	int work_items_per_group = col_count > 256 ? 512 : 256; // Max dimensions 512 for our experiments, but more efficiency at 256
	int k = 5; // get k smallest element

	// allocate memory and copy data
	MGPU_MEM(float) devPtrTest = context->Malloc<float>(test_points, test_count*col_count);
	MGPU_MEM(float) devPtrTrain = context->Malloc<float>(train_points, train_count*col_count);
	MGPU_MEM(float) devPtrOutput = context->Malloc<float>(train_count*test_count);

	// create two dimensional blocks
	dim3 block_size;
	block_size.x = 1;
	block_size.y = work_items_per_group;

	// configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = test_count / block_size.x;
	grid_size.y = train_count / block_size.y;

	int temp_mem = sizeof(float) * col_count; // allocate enough for one training point

	double GPUDistanceTime = 0;

	context->Start();    
	distances_computation <<< grid_size, block_size, temp_mem >>>( devPtrTest->get(),
		devPtrTrain->get(),
		devPtrOutput->get(),
		col_count
		);
	GPUDistanceTime = context->Split();

	MGPU_SYNC_CHECK("distances_computation");

	cout << "GPU Distance Computation Time: " << GPUDistanceTime << "\n";

	// don't need these 2 anymore
	devPtrTest.release();
	devPtrTrain.release();
	// Keep the test and train points so that CPU processing can also happen

	// STAGE 2 BEGIN
	double GPUSortTime = 0;
	
	// Allocate memory for sorting stage
	MGPU_MEM(int) keys = context->Malloc<int>(train_count*test_count);
	MGPU_MEM(int) segments = context->Malloc<int>(test_count);

	// fill in keys here
	context->Start();
	populate_keys_with_training_ids<<< grid_size, block_size >>>(keys->get());
	GPUSortTime = context->Split();

	MGPU_SYNC_CHECK("populate_keys_with_training_ids");

	// fill in segments here

	// create two dimensional blocks
	block_size.x = work_items_per_group;
	block_size.y = 1;

	// configure a two dimensional grid as well
	grid_size.x = test_count / block_size.x;
	grid_size.y = 1;

	context->Start();
	populate_segments<<< block_size, grid_size >>>(segments->get(), train_count);
	GPUSortTime += context->Split();

	context->Start();
	SegSortPairsFromIndices<float, int>(devPtrOutput->get(), keys->get(), train_count*test_count,
		segments->get(), test_count, *context);
	GPUSortTime += context->Split();

	cout << "GPU Sorting Time: " << GPUSortTime << "\n";

	cout << "GPU Time Taken: " << GPUSortTime+GPUDistanceTime << "\n\n";

	int *kSmallestIndicesGPU = new int[test_count*k];

	int offset = 0;

	for(int i=0; i < test_count; i++) {
		keys->ToHost(offset, sizeof(int)*k, &kSmallestIndicesGPU[k*i]);
		offset += sizeof(int)*train_count; // increment to next test point
	}

	// Release data on GPU
	keys.release();
	segments.release();
	devPtrOutput.release();
	// don't need these 2 anymore
	devPtrTest.release();
	devPtrTrain.release();
	// Stage 3 NOT IMPLEMENTED

	// CPU Processing BEGIN

	// Allocate memory for outputpu
	vector<pair<float, int>> CPUOutput(test_count*train_count); // create vector so that it can be easily sorted later

	double CPUDistanceTime = 0;

	auto start_time = chrono::steady_clock::now();

	// Stage 1
	for (int i=0; i < test_count; i++) {
		for (int j=0; j < train_count; j++) {
			float res = 0;
			for (int p=0; p < col_count; p++) {
				res += pow(test_points[i*col_count+p]-train_points[p*train_count+j], 2);
			}

			CPUOutput[i*train_count+j] = pair<float, int>(res, j);
		}
	}

	CPUDistanceTime = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start_time).count() / 1000000.0;
	cout << "CPU Distance Computation Time: " << CPUDistanceTime << "\n";

	// Stage 2
	// Sort each list
	double CPUSortTime = 0;
	start_time = chrono::steady_clock::now();

	for (int i=0; i < test_count; i++) {
		sort(CPUOutput.begin() + i*train_count, CPUOutput.begin() + (i+1)*train_count);
	}
	CPUSortTime = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start_time).count() / 1000000.0;
	cout << "CPU Sorting Time: " << CPUSortTime << "\n";
	cout << "CPU Time Taken: " << CPUDistanceTime+CPUSortTime << "\n";

	// Create a separate small array for the smallest indices
	int* kSmallestIndicesCPU = new int[k*test_count];
	offset = 0;

	start_time = chrono::steady_clock::now();

	for(int i=0; i < test_count; i++) {
		for (int j=0; j < k; j++) {
			kSmallestIndicesCPU[k*i+j] = CPUOutput[i*train_count+j].second;
		}
	}

	// Stage 3 NOT IMPLEMENTED

	// Clear up
	free(test_points);
	free(train_points);

	int rtrn = 0;

	// Selection successful, now check answers against GPU
	for (int i=0; i < test_count*k; i++) {
		if (kSmallestIndicesCPU[i] != kSmallestIndicesGPU[i]) {
			cerr << "ERROR, mismatch at: " << i << "\n";
			rtrn = 1;
		}
	}

	return rtrn; // exit code
}