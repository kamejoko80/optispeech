#!/usr/bin/env python3
import argparse
import numpy as np
import onnx
from onnx import numpy_helper, helper


def add_init(graph, name, array: np.ndarray):
    for it in graph.initializer:
        if it.name == name:
            return
    graph.initializer.append(numpy_helper.from_array(array, name=name))


def add_scalar_i64_init(graph, name, value):
    add_init(graph, name, np.array(value, dtype=np.int64))


def add_i64_vec_init(graph, name, values):
    add_init(graph, name, np.asarray(values, dtype=np.int64))


def add_f32_mat_init(graph, name, values_2d):
    add_init(graph, name, np.asarray(values_2d, dtype=np.float32))


def make_ceil_replacement(inp_name: str, out_name: str, prefix: str):
    n_cast_i64 = helper.make_node(
        "Cast",
        inputs=[inp_name],
        outputs=[f"{prefix}_cast_i64"],
        to=onnx.TensorProto.INT64,
        name=f"{prefix}/CastToI64",
    )
    n_cast_f = helper.make_node(
        "Cast",
        inputs=[f"{prefix}_cast_i64"],
        outputs=[f"{prefix}_cast_back_f"],
        to=onnx.TensorProto.FLOAT,
        name=f"{prefix}/CastToF",
    )
    n_less = helper.make_node(
        "Less",
        inputs=[f"{prefix}_cast_back_f", inp_name],
        outputs=[f"{prefix}_less"],
        name=f"{prefix}/Less",
    )
    n_cast_add = helper.make_node(
        "Cast",
        inputs=[f"{prefix}_less"],
        outputs=[f"{prefix}_add_i64"],
        to=onnx.TensorProto.INT64,
        name=f"{prefix}/CastAdd",
    )
    n_add = helper.make_node(
        "Add",
        inputs=[f"{prefix}_cast_i64", f"{prefix}_add_i64"],
        outputs=[f"{prefix}_ceil_i64"],
        name=f"{prefix}/Add",
    )
    n_out = helper.make_node(
        "Cast",
        inputs=[f"{prefix}_ceil_i64"],
        outputs=[out_name],
        to=onnx.TensorProto.FLOAT,
        name=f"{prefix}/CastOutF",
    )
    return [n_cast_i64, n_cast_f, n_less, n_cast_add, n_add, n_out]


def make_cumsum_replacement(inp_name: str, out_name: str, T: int, graph, prefix: str):
    # Input to CumSum here is actually [B, T] (you saw {2,64}).
    # Reshape to [0, T] means "copy dim0 from input" => [B, T] always valid.
    shape_name = f"{prefix}__shape_0_T__"
    U_name = f"{prefix}__U_triu_ones__"

    add_i64_vec_init(graph, shape_name, [0, T])

    U = np.triu(np.ones((T, T), dtype=np.float32))
    add_f32_mat_init(graph, U_name, U)

    n_reshape = helper.make_node(
        "Reshape",
        inputs=[inp_name, shape_name],
        outputs=[f"{prefix}_reshape"],
        name=f"{prefix}/Reshape",
    )
    n_cast = helper.make_node(
        "Cast",
        inputs=[f"{prefix}_reshape"],
        outputs=[f"{prefix}_xf"],
        to=onnx.TensorProto.FLOAT,
        name=f"{prefix}/CastToF",
    )
    n_mm = helper.make_node(
        "MatMul",
        inputs=[f"{prefix}_xf", U_name],
        outputs=[f"{prefix}_y"],
        name=f"{prefix}/MatMul",
    )
    n_out = helper.make_node(
        "Cast",
        inputs=[f"{prefix}_y"],
        outputs=[out_name],
        to=onnx.TensorProto.INT64,
        name=f"{prefix}/CastOutI64",
    )
    return [n_reshape, n_cast, n_mm, n_out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--frames_per_token", type=int, default=20)
    ap.add_argument("--hop", type=int, default=256)
    args = ap.parse_args()

    T = int(args.T)
    F = int(T * args.frames_per_token)
    S = int(F * args.hop)

    caps = {
        "/Range": T,
        "/text_embedding/embed_positions/Range": T,
        "/decoder/pos_emb/Range": S,
        "/feature_upsampler/Range": S,
        "/Range_1": S,
    }

    m = onnx.load(args.inp)

    patched_range = 0
    patched_ceil = 0
    patched_cumsum = 0

    new_nodes = []
    for node in m.graph.node:
        if node.op_type == "Range" and node.name in caps:
            cap = caps[node.name]
            init_name = f"__cap_{node.name.replace('/', '_')}_{cap}__"
            add_scalar_i64_init(m.graph, init_name, cap)
            node.input[1] = init_name
            patched_range += 1
            new_nodes.append(node)
            continue

        if node.op_type == "Ceil" and node.name == "/Ceil":
            inp = node.input[0]
            out = node.output[0]
            new_nodes.extend(make_ceil_replacement(inp, out, prefix="__patch_ceil__"))
            patched_ceil += 1
            continue

        if node.op_type == "CumSum" and node.name == "/feature_upsampler/CumSum":
            inp = node.input[0]
            out = node.output[0]
            new_nodes.extend(make_cumsum_replacement(inp, out, T=T, graph=m.graph, prefix="__patch_cumsum__"))
            patched_cumsum += 1
            continue

        new_nodes.append(node)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    onnx.checker.check_model(m)
    onnx.save(m, args.out)

    print("Patched Range nodes:", patched_range)
    print("Patched Ceil nodes:", patched_ceil)
    print("Patched CumSum nodes:", patched_cumsum)
    print("Caps: T =", T, "F =", F, "S =", S)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
