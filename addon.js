var addon = require('bindings')('ml');

var obj = new addon.NodePersona(10);
console.log( obj.plusOne() ); // 11
console.log( obj.plusOne() ); // 12
console.log( obj.plusOne() ); // 13

obj.train("1", function(message) {
  console.log(message);
});