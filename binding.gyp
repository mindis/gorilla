{
  "targets": [
    {
      "target_name": "ml",
      "sources": ["ml.cc", "nodeliblinear.cc"],
      "include_dirs": [],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions" ],
      "cflags!": [ '-Wall', '-O3', '-c', "-fno-exceptions"],
    }
  ]
}
