#ifndef NODELIBLINEAR_H
#define NODELIBLINEAR_H

#include <node.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct DataT
{
	int num_train;
	int num_test;
	int num_validation;
	int num;
	int curr;
	int* train_ids;
	int* test_ids;
	int* validation_ids;
	float** x;
	float* c;
	float* y;
	float* yt;

	// intermediate variables for solver
    float* QD;
    float* alpha;
};

typedef struct DataT* GorillaData;

struct LinearModelT
{
	float* w;  // personalized model
	float* z;  // consensus model
    float* mu; // dual
    float* q;  // local copy of the consensus model
    float* dq; // difference of the local consensus model

    float gamma;
    float rho;
    float rho0;
    float eps;
    int max_num_iters;
};

typedef struct LinearModelT* GorillaModel;

class NodeLiblinear : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> target);

 private:
  int _dim;
  FILE* fp;
  GorillaData data;
  GorillaModel model;
 
  void _zeroA(float* a);
  void _copyAB(float* a, float* b);
  void _addAB(float* a, float* b);
  void _subAB(float* a, float* b);
  void _addABW(float* a, float* b, float w);
  void _scaleA(float* a, float w);
  float _dotAB(float* a, float* b);
  float _normA(float* a);
  void _swap(int* a, int i, int j);
  void write_float(float value);
  float read_float();
  void write_int(int value);
  int read_int();
  void write_array(float* a, int dim);
  float* read_array(int dim);
  void write_array_int(int* a, int dim);
  int* read_array_int(int dim);
  void write_matrix(float** matrix, int num, int dim);
  float** read_matrix(int num, int dim);
  void printA(int* a, int dim);

  void serialize(char* filename);
  void deserialize(char* filename);

  void set_dim(int dim);

  int create_data(int num);
  void destroy_data();
  int is_data_ready();
  void split_data(float ratio_train, float ratio_test);
  void add_data(float* x, float y, float c);
  void random_data();

  int create_model(float gamma, float rho, float eps, int max_num_iters);
  void destroy_model();
  int is_model_ready();
  void reset_model_parameter(float gamma, float rho, float eps, int max_num_iters);
  void reset_model_values();

  void init_solver(int warm_start);
  void solve();

  void train(float* z, int warm_start); // consensus from parent
  void merge(float* dq); // local difference
  float predict(float* x);
  void validate();
  void test();

 private:
  NodeLiblinear();
  ~NodeLiblinear();

  static v8::Handle<v8::Value> New(const v8::Arguments& args);
  static v8::Handle<v8::Value> PlusOne(const v8::Arguments& args);
  static v8::Handle<v8::Value> TrainUser(const v8::Arguments& args);

  int counter_;
};

#endif
