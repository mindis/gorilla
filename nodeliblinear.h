#ifndef NODELIBLINEAR_H
#define NODELIBLINEAR_H

#include <node.h>

class NodeLiblinear : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> target);

 private:
  NodeLiblinear();
  ~NodeLiblinear();

  static v8::Handle<v8::Value> New(const v8::Arguments& args);
  static v8::Handle<v8::Value> PlusOne(const v8::Arguments& args);
  static v8::Handle<v8::Value> TrainUser(const v8::Arguments& args);
  double counter_;
};

#endif
