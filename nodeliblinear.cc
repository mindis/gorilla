#include <node.h>
#include "nodeliblinear.h"

using namespace v8;

extern "C" {
  int main(int argc, char **argv);
  void train_heart_scale();
}

NodeLiblinear::NodeLiblinear() {};
NodeLiblinear::~NodeLiblinear() {};

void NodeLiblinear::Init(Handle<Object> target) {
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(String::NewSymbol("NodeLiblinear"));
  
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
  target->Set(String::NewSymbol("NodeLiblinear"), constructor);
}

Handle<Value> NodeLiblinear::New(const Arguments& args) {
  HandleScope scope;

  NodeLiblinear* obj = new NodeLiblinear();
  obj->counter_ = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
  obj->Wrap(args.This());

  return args.This();
}

/* A test entry-point 
 */
Handle<Value> NodeLiblinear::PlusOne(const Arguments& args) {
  HandleScope scope;

  NodeLiblinear* obj = ObjectWrap::Unwrap<NodeLiblinear>(args.This());
  obj->counter_ += 1;

  return scope.Close(Number::New(obj->counter_));
}

Handle<Value> NodeLiblinear::TrainUser(const Arguments& args) {
  HandleScope scope;

  NodeLiblinear* obj = ObjectWrap::Unwrap<NodeLiblinear>(args.This()); 
  
  // Get user id
  Local<String> uid = args[0]->ToString();

  // Train
  char* options[] = {
    "train",
    "-s",
    "3",
    "heart_scale",
    "tmp.model"
  };
  main(5, options);
  // train_heart_scale();

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