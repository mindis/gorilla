{
  "targets": [
    {
      "target_name": "ml",
      "sources": ["ml.cc", "nodeliblinear.cc", "linear.cc", "train.c", "train.cc", "tron.cc"],
      "include_dirs": ["blas"],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions" ],
      "cflags!": [ '-Wall', '-O3', '-fPIC', '-c', "-fno-exceptions"],
    }
  ]
}
