var express = require('express'),
  redis = require('redis'),
  ml = requires('ml'),
  model = ml.NodeLiblinear(10),
  client = redis.createClient(),
  app = express();

client.on("message", function (channel, message) {
  var user_id = channel;
  if (message == "train") {
    var localModel = model.train(user_id);
  } else {

  }
});

// Start a server
app.get('/start', function (req, res) {

      // // Receiving Event
      // client.on("message", function (channel, message) {

      // });

      // // client should be subscribing to it owns childs
      // client.subscribe("channel1");
  
  res.sendStatus(200);
});

// Register a user
app.get('/users/register', function (req, res)) {
  var user_id = req.params['user_id'];

  model.register(user_id);

  client.subscribe(user_id);
}


// Start Server
var port  = process.env.PORT || process.argv[2] || 18090;
app.listen(port);
console.log("Listening on port " + port);