{
  "targets": [
    {
      "target_name": "ml",
      "sources": ["ml.cc", "nodepersona.cc"],
      "include_dirs": [],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions" ],
      "cflags!": [ '-Wall', '-O3', '-c', "-fno-exceptions"],
    }
  ]
}
