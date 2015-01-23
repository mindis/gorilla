#include <node.h>
#include "nodepersona.h"

using namespace v8;

void InitAll(Handle<Object> exports) {
    NodePersona::Init(exports);
}

NODE_MODULE(ml, InitAll)
