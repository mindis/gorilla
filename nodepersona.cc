#include <node.h>
#include "nodepersona.h"

#define _freeA(a) if ((a) != 0) {free(a); a = 0;}
#define _max(a, b) ((a)>(b))?(a):(b);
#define _min(a, b) ((a)<(b))?(a):(b);

using namespace v8;

NodePersona::NodePersona() {};
NodePersona::~NodePersona() {};

void NodePersona::_zeroA(float* a)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] = 0;
  }
}

void NodePersona::_copyAB(float* a, float* b)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] = b[i];
  }
}

void NodePersona::_addAB(float* a, float* b)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] += b[i];
  }
}

void NodePersona::_subAB(float* a, float* b)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] -= b[i];
  }
}

void NodePersona::_addABW(float* a, float* b, float w)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] += b[i] * w;
  }
}

void NodePersona::_scaleA(float* a, float w)
{
  int i;
  for (i = 0; i < _dim; i++)
  {
    a[i] *= w;
  }
}

float NodePersona::_dotAB(float* a, float* b)
{
  int i;
  float result = 0;
  for (i = 0; i < _dim; i++)
  {
    result += a[i] * b[i];
  }
  return result;
}

float NodePersona::_normA(float* a)
{
  return _dotAB(a, a);
}

void NodePersona::_swap(int* a, int i, int j)
{
  int tmp = a[i];
  a[i] = a[j];
  a[j] = tmp;
}

void NodePersona::write_float(float value)
{
  fprintf(fp, "%f\n", value);
}

float NodePersona::read_float()
{
  float value;
  fscanf(fp, "%f\n", &value);
  return value;
}

void NodePersona::write_int(int value)
{
  fprintf(fp, "%d\n", value);
}

int NodePersona::read_int()
{
  int value;
  fscanf(fp, "%d\n", &value);
  return value;
}

void NodePersona::write_array(float* a, int dim)
{
  int i;
  if (dim <= 0)
    return;

  for (i = 0; i < dim - 1; i++)
  {
    fprintf(fp, "%f,", a[i]);
  }
  fprintf(fp, "%f\n", a[i]);
}

float* NodePersona::read_array(int dim)
{
  int i;
  if (dim <= 0)
    return 0;

  float* a = (float*)malloc(dim*sizeof(float));
  float* ap = a;
  for (i = 0; i < dim - 1; i++)
  {
    fscanf(fp, "%f,", ap++);
  }
  fscanf(fp, "%f\n", ap++);
  return a;
}

void NodePersona::write_array_int(int* a, int dim)
{
  if (dim <= 0)
    return;

  int i;
  for (i = 0; i < dim - 1; i++)
  {
    fprintf(fp, "%d,", a[i]);
  }
  fprintf(fp, "%d\n", a[i]);
}

int* NodePersona::read_array_int(int dim)
{
  int i;
  if (dim <= 0)
    return 0;

  int* a = (int*)malloc(dim*sizeof(int));
  int* ap = a;
  for (i = 0; i < dim - 1; i++)
  {
    fscanf(fp, "%d,", ap++);
  }
  fscanf(fp, "%d\n", ap++);
  return a;
}

void NodePersona::write_matrix(float** matrix, int num, int dim)
{
  int i;
  if (dim <= 0 || num <= 0)
    return;

  for (i = 0; i < num; i++)
  {
    write_array(matrix[i], dim);
  }
}

float** NodePersona::read_matrix(int num, int dim)
{
  int i;
  if (dim <= 0 || num <= 0)
    return 0;

  float** a = (float**)malloc(num*sizeof(float*));
  for (i = 0; i < num; i++)
  {
    a[i] = read_array(dim);
  }
  return a;
}

void NodePersona::serialize(char* filename)
{
  fp = fopen(filename, "w");
  // dimension
  write_int(_dim);
  // data
  write_int(data->num_train);
  write_int(data->num_test);
  write_int(data->num_validation);
  write_int(data->num);
  write_int(data->curr);
  write_array_int(data->train_ids, data->num_train);
  write_array_int(data->test_ids, data->num_test);
  write_array_int(data->validation_ids, data->num_validation);
  write_matrix(data->x, data->num, _dim);
  write_array(data->c, data->num);
  write_array(data->y, data->num);
  write_array(data->yt, data->num);
  write_array(data->QD, data->num);
  write_array(data->alpha, data->num);
  // model
  write_int(model->max_num_iters);
  write_float(model->eps);
  write_float(model->rho0);
  write_float(model->rho);
  write_float(model->gamma);
  write_array(model->w, _dim);
  write_array(model->z, _dim);
  write_array(model->mu, _dim);
  write_array(model->q, _dim);
  write_array(model->dq, _dim);
  fclose(fp);
  fp = 0;
}

void NodePersona::deserialize(char* filename)
{
  fp = fopen(filename, "r");
  // dimension
  _dim = read_int();
  // data
  data = (GorillaData)malloc(sizeof(struct DataT));
  data->num_train = read_int();
  data->num_test = read_int();
  data->num_validation = read_int();
  data->num = read_int();
  data->curr = read_int();
  data->train_ids = read_array_int(data->num_train);
  data->test_ids = read_array_int(data->num_test);
  data->validation_ids = read_array_int(data->num_validation);
  data->x = read_matrix(data->num, _dim);
  data->c = read_array(data->num);
  data->y = read_array(data->num);
  data->yt = read_array(data->num);
  data->QD = read_array(data->num);
  data->alpha = read_array(data->num);
  // model
  model = (GorillaModel)malloc(sizeof(struct LinearModelT));
  model->max_num_iters = read_int();
  model->eps = read_float();
  model->rho0 = read_float();
  model->rho = read_float();
  model->gamma = read_float();
  model->w = read_array(_dim);
  model->z = read_array(_dim);
  model->mu = read_array(_dim);
  model->q = read_array(_dim);
  model->dq = read_array(_dim);
  fclose(fp);
  fp = 0;
}

void NodePersona::set_dim(int dim)
{
  _dim = dim;
}

int NodePersona::create_data(int num)
{
  int i;
  if (data != 0) return -1;
  data = (GorillaData)malloc(sizeof(struct DataT));
  data->num = num;
  data->num_train = 0;
  data->num_test = 0;
  data->num_validation = 0;
  data->curr = 0;
  data->train_ids = 0;
  data->test_ids = 0;
  data->validation_ids = 0;
  data->x = (float**)malloc(num*sizeof(float*));
  for (i = 0; i < num; i++) 
  {
    data->x[i] = (float*)calloc(_dim, sizeof(float));
  }
  data->y  = (float*)calloc(num, sizeof(float));
  data->yt = (float*)calloc(num, sizeof(float));
  data->c  = (float*)calloc(num, sizeof(float));
  for (i = 0; i < num; i++)
  {
    data->c[i] = 1.0;
  }
  data->QD = (float*)calloc(num, sizeof(float));
  data->alpha = (float*)calloc(num, sizeof(float));
  return 1;
}

void NodePersona::destroy_data()
{
  int i;
  if (data != 0)
  {
    _freeA(data->train_ids);
    _freeA(data->test_ids);
    _freeA(data->validation_ids);
    _freeA(data->y);
    _freeA(data->yt);
    _freeA(data->c);
    _freeA(data->QD);
    _freeA(data->alpha);
    for (i = 0; i < data->num; i++)
    {
      _freeA(data->x[i]);
    }
    _freeA(data->x);
    free(data);
    data = 0;
  }
}

int NodePersona::is_data_ready()
{
  return (data != 0);
}

void NodePersona::printA(int* a, int dim)
{
  int i;
  for (i = 0; i < dim-1; i++)
  {
    printf("%d, ", a[i]);
  }
  printf("%d\n", a[i]);
}

void NodePersona::split_data(float ratio_train, float ratio_test)
{
  int* pos_index = (int*)malloc(data->num*sizeof(int));
  int* neg_index = (int*)malloc(data->num*sizeof(int));
  int i, k, pi=0, ni=0;
  int j, pitrain, pitest, nitrain, nitest, pivalidation;

  for (i = 0; i < data->num; i++)
  {
    if (data->y[i] > 0)
    {
      pos_index[pi++] = i;
    }
    else
    {
      neg_index[ni++] = i;
    }
  }

  //printA(pos_index, pi);
  for (i = 0; i < pi; i++)
  {
    k = i + rand() % (pi - i);
    _swap(pos_index, i, k);
  }
  //printA(pos_index, pi);

  //printA(neg_index, ni);
  for (i = 0; i < ni; i++)
  {
    k = i + rand() % (ni - i);
    _swap(neg_index, i, k);
  }
  //printA(neg_index, ni);

  pitrain = (int)(ratio_train*pi);
  nitrain = (int)(ratio_train*ni);
  pitest = (int)(ratio_test*pi);
  nitest = (int)(ratio_test*ni);
  data->num_train = pitrain + nitrain;
  data->num_test = pitest + nitest;
  data->num_validation = data->num - data->num_train - data->num_test;
  _freeA(data->train_ids);
  _freeA(data->test_ids);
  _freeA(data->validation_ids);
  data->train_ids = (int*)malloc(data->num_train*sizeof(int));
  data->test_ids = (int*)malloc(data->num_test*sizeof(int));
  data->validation_ids = (int*)malloc(data->num_validation*sizeof(int));
  j = 0;
  for (i = 0; i < pitrain; i++,j++)
  {
    data->train_ids[i] = pos_index[j];
  }
  //printA(data->train_ids, pitrain);
  for (i = 0; i < pitest; i++,j++)
  {
    data->test_ids[i] = pos_index[j];
  }
  //printA(data->test_ids, pitest);
  for (i = 0; j < pi; i++,j++)
  {
    data->validation_ids[i] = pos_index[j];
  }
  pivalidation = i;
  //printA(data->validation_ids, pivalidation);

  j = 0;
  for (i = 0; i < nitrain; i++,j++)
  {
    data->train_ids[pitrain + i] = neg_index[j];
  }
  //printA(data->train_ids, pitrain + nitrain);
  for (i = 0; i < nitest; i++,j++)
  {
    data->test_ids[pitest + i] = neg_index[j];
  }
  //printA(data->test_ids, pitest + nitest);
  for (i = 0; j < ni; i++,j++)
  {
    data->validation_ids[pivalidation + i] = neg_index[j];
  }
  //printA(data->validation_ids, pivalidation + i);
  _freeA(pos_index);
  _freeA(neg_index);
} 

void NodePersona::random_data()
{
  int i,j;
  for (i = 0; i < data->num; i++)
  {
    for (j = 0; j < _dim; j++)
    {
      data->x[i][j] = (float)(rand()) / RAND_MAX;
    }
    data->y[i] = ((float)(rand()) / RAND_MAX > 0.5)?(1.0):(-1.0);
  }
}

void NodePersona::add_data(float* x, float y, float c)
{
  int i = 0;
  float* currX = data->x[data->curr];
  for (i = 0; i < _dim; i++)
  {
    currX[i] = x[i];
  }
  data->y[data->curr] = y;
  data->c[data->curr] = c;

  data->curr++;
  if (data->curr >= data->num)
  {
    data->curr = 0;
  }
}

int NodePersona::create_model(float gamma, float rho, float eps, int max_num_iters)
{
  if (model != 0) return -1;
  model = (GorillaModel)malloc(sizeof(struct LinearModelT));
  model->gamma = gamma;
  model->rho = rho;
  model->eps = eps;
  model->rho0 = rho * gamma / (2 * (rho + gamma));
  model->max_num_iters = max_num_iters;
  model->w  = (float*)calloc(_dim, sizeof(float));
  model->z  = (float*)calloc(_dim, sizeof(float));
  model->mu = (float*)calloc(_dim, sizeof(float));
  model->q  = (float*)calloc(_dim, sizeof(float));
  model->dq = (float*)calloc(_dim, sizeof(float));
  return 1;
}

void NodePersona::destroy_model()
{
  if (model != 0)
  {
    _freeA(model->dq);
    _freeA(model->q);
    _freeA(model->mu);
    _freeA(model->z);
    _freeA(model->w);
    free(model);
    model = 0;
  }
}

int NodePersona::is_model_ready()
{
  return (model!=0);
}

void NodePersona::reset_model_parameter(float gamma, float rho, float eps, int max_num_iters)
{
  if (is_model_ready() > 0)
  {
    model->gamma = gamma;
    model->rho = rho;
    model->rho0 = rho * gamma / (2 * (rho + gamma));
    model->max_num_iters = max_num_iters;
    model->eps = eps;
  }
}

void NodePersona::reset_model_values()
{
  if (is_model_ready() > 0)
  {
    _zeroA(model->w);
    _zeroA(model->z);
    _zeroA(model->mu);
    _zeroA(model->q);
    _zeroA(model->dq);
  }
}

void NodePersona::init_solver(int warm_start)
{
  int i, j;
  float* w = model->w;
  _copyAB(w, model->z);
  if (warm_start > 0)
  {
    for (i = 0; i < data->num_train; i++)
    {
      j = data->train_ids[i];
      _addABW(w, data->x[j], data->y[j] * data->alpha[j]);
    }
  }
  else
  {
    for (i = 0; i < data->num_train; i++)
    {
      j = data->train_ids[i];
      data->QD[j] = model->rho0 / data->c[j] + _normA(data->x[j]);
      data->alpha[j] = 0;
    }
  }
  _subAB(w, model->mu);
}

void NodePersona::solve()
{
  int j, k, s, iteration;
  float d, G, alpha_old;

  int active_size = data->num_train;
  int* index = data->train_ids;
  float* alpha = data->alpha;
  float* QD = data->QD;
  float* c = data->c;
  
  int yj;
  float* xj;

  float* w = model->w;
  float rho0 = model->rho0;
  float eps = model->eps;
  int max_num_iters = model->max_num_iters;

  // PG: projected gradient, for shrinking and stopping
  float PG;
  float PGmax_old = 1e10;
  float PGmin_old = -1e10;
  float PGmax_new;
  float PGmin_new;

  iteration = 0;
  while (iteration < max_num_iters)
  {
      PGmax_new = -1e10;
      PGmin_new = 1e10;
      for (j = 0; j < active_size; j++)
      {
        k = j + rand() % (active_size - j);
          _swap(index, j, k);
      }
          
      for (s = 0; s < active_size; s++)
      {
          j  = index[s];
          yj = data->y[j];
          xj = data->x[j];
          
          G = _dotAB(w, xj) * yj - 1;
          G += alpha[j] * rho0 / c[j];
          
          PG = 0;
          if (alpha[j] <= 0)
          {
              if (G > PGmax_old)
              {
                  active_size -= 1;
                  _swap(index, s, active_size);
                  s -= 1;
                  continue;
              }
              else if (G < 0)
              {
                  PG = G;
              }
          }
          else
          {
              PG = G;
          }

          PGmax_new = _max(PGmax_new, PG);
          PGmin_new = _min(PGmin_new, PG);
          
          if (PG > 1e-12 || PG < -1e-12)
          {
              alpha_old = alpha[j];
              alpha[j] = _max(alpha[j] - G / QD[j], 0);
              d = (alpha[j] - alpha_old) * yj;
              _addABW(w, xj, d);
          }
      }
                  
      iteration += 1;

      if (PGmax_new - PGmin_new <= eps)
      {
          if (active_size == data->num_train)
          {
              break;              
          }
          else
          {
      active_size = data->num_train;
              PGmax_old = 1e10;
              PGmin_old = -1e10;
              continue;             
          }    
      }
          
      PGmax_old = PGmax_new;
      PGmin_old = PGmin_new;
      if (PGmax_old <= 0)
          PGmax_old = 1e10;
      if (PGmin_old >= 0)
          PGmin_old = -1e10;
  }
}

void NodePersona::train(float* z, int warm_start)
{
  float r;
  float* dq = model->dq;
  float* q = model->q;
  float* w = model->w;
  float* mu = model->mu;

  if (z != 0)
  {
    // copy to model
    _copyAB(model->z, z);
  }
  else
  {
    z = model->z;
  }

  // update mu
  _zeroA(dq);
  _subAB(dq, mu);
  _addAB(mu, q);
  _subAB(mu, z);
  _addAB(dq, mu);

  //metricAbs(self, 'mu', model->mu)
  //metricAbs(metricLog, self, '|dmu|', model->dq)
  //metricValue(metricLog, self, 'sup(mu)', 2 * model->solver.num_instances * model->solver.maxxnorm() * z.norm())

  // update w
  init_solver(warm_start);
  solve();
  //loss = model->solver.status()
  //metricValue(metricLog, self, 'loss', loss)
  //metricRelAbs(metricLog, self, '|q~w|', model->q, model->panda.weights)
  //loss = model->solver.status()
  //metricValue(self, 'loss', loss)
  //metricValue(self, 'x', model->solver.maxxnorm())

  // update q
  r = model->rho / (model->rho + model->gamma);
  _subAB(dq, q);
  _zeroA(q);
  _addABW(q, z, r);
  _addABW(q, w, 1-r);
  _addABW(q, mu, -r);
  _addAB(dq, q);
}

void NodePersona::merge(float* dq)
{
  //rd = (fdq.norm() + EPS) / (self.panda.z.norm() + EPS)
  //rd < eps --> converged
  _addABW(model->z, dq, 1.0 / (data->num_train + 1.0 / model->rho));
}

float NodePersona::predict(float* x)
{
  return _dotAB(model->w, x);
}

void NodePersona::validate()
{
  int i, j;
  for (i = 0; i < data->num_validation; i++)
  {
    j = data->validation_ids[i];
    data->yt[j] = predict(data->x[j]);
  }
}

void NodePersona::test()
{
  int i, j;
  for (i = 0; i < data->num_test; i++)
  {
    j = data->test_ids[i];
    data->yt[j] = predict(data->x[j]);
  }
}

/**
 ***************************************************************************************Static Wrap up Functions
 */

/**
 * Initialize data and model at the first place, assuming each instance is for one user
 * Input : int dim, the number of features to use
 *         int num, the number of data that belong to this user
 *         float gamma, the strength of the personality. [0, infinity], 0 means completely a lone wolf, infinity means completely a sheep. usually 1
 *         float rho, the parameter for ADMMs, usually 1.
 *         float eps, the precision tolerance. 1e-4 should be good enough
 *         int max_num_iters, the maximum number of iterations for dual coordinate descent solver can run
 *         char* cache_file_name, the file name to take a snapshot of this module
 */
void NodePersona::Init(Handle<Object> target) {
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(String::NewSymbol("NodePersona"));
  
  // How many fields it contains
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  
  // plusOne -> static PlusOne
  tpl->PrototypeTemplate()->Set(String::NewSymbol("plusOne"),
      FunctionTemplate::New(PlusOne)->GetFunction());
  // train -> TrainUser
  tpl->PrototypeTemplate()->Set(String::NewSymbol("train"),
      FunctionTemplate::New(TrainUser)->GetFunction());

  // Set up constructor
  Persistent<Function> constructor = Persistent<Function>::New(tpl->GetFunction());
  target->Set(String::NewSymbol("NodePersona"), constructor);
}

/**
 * Reset the parameters for the model
 * Input: 
 *         float gamma, the strength of the personality. [0, infinity], 0 means completely a lone wolf, infinity means completely a sheep. usually 1
 *         float rho, the parameter for ADMMs, usually 1.
 *         float eps, the precision tolerance. 1e-4 should be good enough
 *         int max_num_iters, the maximum number of iterations for dual coordinate descent solver can run
 */
void NodePersona::ResetParameter(const Arguments& args)
{
  //TODO:
}

/**
 * Reset the coefficients for the model
 */
void NodePersona::ResetModel(const Arguments& args)
{
  //TODO:
}

/**
 * Does one local optimization
 * Input: float[] z, a consensus model passed from parent
 *        int warm_start, a flag to choose warm start or not
 */
void NodePersona::Train(const Arguments& args)
{
  //TODO:
}

/**
 * Does one merge step for Asynchronous ADMMs
 * Input: float[] dq, a difference from the child personal model
 */
void NodePersona::Merge(const Arguments& args)
{
  //TODO:
}

/**
 * Perform prediction for the validation dataset
 */
void NodePersona::Validate(const Arguments& args)
{
  //TODO:
}

/**
 * Perform prediction for the test dataset
 */
void NodePersona::Test(const Arguments& args)
{
  //TODO:
}

/**
 * Add one data for this personal model
 * Input: float[] x, an array of feature values
 *        float y, the target value (-1 for negative example, 1 for positive example)
 *        float c, the weight to this example, (might have different value for positive example, and negative example)
 */
void NodePersona::AddData(const Arguments& args)
{
  //TODO:
}

/**
 * Perform a stratified sampling over the data for this user
 * Input: float ratio_train, the fraction for the training data
 *        float ratio_test, the fraction for the test data
 */
void NodePersona::SplitData(const Arguments& args)
{
  //TODO:
}

/**
 * Create a random data for debuging purpose
 */
void NodePersona::RandomData(const Arguments& args)
{
  //TODO:
}

/**
 * Taking a snapshot of the model and the data, store them in a file
 */
void NodePersona::Serialize(const Arguments& args)
{
  //TODO:
}

/**
 * Restore the object from the cache file
 */
void NodePersona::Deserialize(const Arguments& args)
{
  //TODO:
}

/**
 * After performing test, get the ROC arrays
 * Output: float[], an array of FPR
 *         float[], an array of recalls
 */
Handle<Value> NodePersona::GetROC(const Arguments& args)
{
  //TODO:
}

/**
 * After performing test, get the PRC arrays
 * Output: float[], an array of Precisions
 *         float[], an array of Recalls
 */
Handle<Value> NodePersona::GetPRC(const Arguments& args)
{
  //TODO:
}

/**
 * After performing test, get the accuracy report
 */
Handle<Value> NodePersona::GetAccuracy(const Arguments& args)
{
  //TODO:
}


Handle<Value> NodePersona::New(const Arguments& args) {
  HandleScope scope;

  NodePersona* obj = new NodePersona();
  obj->counter_ = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
  obj->Wrap(args.This());

  return args.This();
}

/* A test entry-point 
 */
Handle<Value> NodePersona::PlusOne(const Arguments& args) {
  HandleScope scope;

  NodePersona* obj = ObjectWrap::Unwrap<NodePersona>(args.This());
  obj->counter_ += 1;

  return scope.Close(Number::New(obj->counter_));
}

Handle<Value> NodePersona::TrainUser(const Arguments& args) {
  HandleScope scope;

  NodePersona* obj = ObjectWrap::Unwrap<NodePersona>(args.This()); 
  
  // Get user id
  Local<String> uid = args[0]->ToString();

  /* Train
  char* options[] = {
    "train",
    "-s",
    "3",
    "heart_scale",
    "tmp.model"
  };
  */
  
  if (args.Length() > 1) {
    // Fetch the callback function
    Local<Function> callback = Local<Function>::Cast(args[1]);

    // Make arguments for the callback function
    const unsigned argc = 1;
    Local<Value> argv[argc] = { Local<Value>::New(String::New("hello node!")) };

    // Invoke
    callback -> Call(Context::GetCurrent()->Global(), argc, argv);
  }

  return scope.Close(Undefined());
}