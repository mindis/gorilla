#include <node.h>
#include "nodeliblinear.h"

using namespace v8;

void InitAll(Handle<Object> exports) {
    NodeLiblinear::Init(exports);
}

NODE_MODULE(ml, InitAll)
