"""Read-only probes of the things that decide whether a model deploys.

* `flatbuffer`      -- parse a .tflite directly (never via the interpreter)
* `tflm_introspect` -- what ops the TFLite-Micro tree can register at all
* `accel_map`       -- which of those have a HiFi4 xa_nnlib kernel in the built lib
* `litert_host`     -- load a model in a subprocess so a SIGSEGV is survivable
"""

__all__ = ["flatbuffer", "tflm_introspect", "accel_map", "litert_host"]
