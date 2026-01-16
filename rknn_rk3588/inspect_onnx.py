import onnx

m = onnx.load("models/onnx/lightspeech/en-us/mike-step_305k.onnx")
print("Inputs:")
for i in m.graph.input:
    t = i.type.tensor_type
    shape = [d.dim_value if d.dim_value > 0 else -1 for d in t.shape.dim]
    dtype = t.elem_type
    print(" ", i.name, "shape=", shape, "dtype_enum=", dtype)

print("Outputs:")
for o in m.graph.output:
    t = o.type.tensor_type
    shape = [d.dim_value if d.dim_value > 0 else -1 for d in t.shape.dim]
    dtype = t.elem_type
    print(" ", o.name, "shape=", shape, "dtype_enum=", dtype)
