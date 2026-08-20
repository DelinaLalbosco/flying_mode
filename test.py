import onnx
import onnxruntime as ort
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "policy.onnx"

print("=" * 70)
print("ONNX MODEL INSPECTION")
print("=" * 70)
print("File:", os.path.abspath(path))
print("Size:", os.path.getsize(path) / (1024 * 1024), "MB")

# ------------------------------------------------------------
# ONNX graph
# ------------------------------------------------------------

model = onnx.load(path)

print("\n[MODEL]")
print("IR version:", model.ir_version)
print("Producer:", model.producer_name)
print("Producer version:", model.producer_version)
print("Domain:", model.domain)
print("Model version:", model.model_version)

print("\n[INPUTS]")
for i, inp in enumerate(model.graph.input):
    shape = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value:
            shape.append(d.dim_value)
        elif d.dim_param:
            shape.append(d.dim_param)
        else:
            shape.append("?")

    print(f"Input {i}:")
    print("  name :", inp.name)
    print("  shape:", shape)

print("\n[OUTPUTS]")
for i, out in enumerate(model.graph.output):
    shape = []
    for d in out.type.tensor_type.shape.dim:
        if d.dim_value:
            shape.append(d.dim_value)
        elif d.dim_param:
            shape.append(d.dim_param)
        else:
            shape.append("?")

    print(f"Output {i}:")
    print("  name :", out.name)
    print("  shape:", shape)

# ------------------------------------------------------------
# Initializers / parameters
# ------------------------------------------------------------

print("\n[PARAMETERS]")

total_parameters = 0

for initializer in model.graph.initializer:
    count = 1
    for d in initializer.dims:
        count *= d

    total_parameters += count

print("Total parameters:", total_parameters)
print("Approx parameter memory:",
      total_parameters * 4 / (1024 * 1024), "MB")

# ------------------------------------------------------------
# Operators
# ------------------------------------------------------------

print("\n[OPERATORS]")

ops = {}

for node in model.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1

for op, count in sorted(ops.items()):
    print(f"  {op:25s}: {count}")

print("\nTotal graph nodes:", len(model.graph.node))

# ------------------------------------------------------------
# ONNX Runtime
# ------------------------------------------------------------

print("\n[ONNX RUNTIME]")

session = ort.InferenceSession(
    path,
    providers=["CPUExecutionProvider"]
)

print("Providers:", session.get_providers())

print("\nRuntime inputs:")
for inp in session.get_inputs():
    print("  name :", inp.name)
    print("  shape:", inp.shape)
    print("  type :", inp.type)

print("\nRuntime outputs:")
for out in session.get_outputs():
    print("  name :", out.name)
    print("  shape:", out.shape)
    print("  type :", out.type)

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

print("\n[METADATA]")

metadata = session.get_modelmeta()

print("Description:", metadata.description)
print("Graph name :", metadata.graph_name)
print("Producer   :", metadata.producer_name)
print("Version    :", metadata.version)

if metadata.custom_metadata_map:
    print("\nCustom metadata:")
    for key, value in metadata.custom_metadata_map.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)