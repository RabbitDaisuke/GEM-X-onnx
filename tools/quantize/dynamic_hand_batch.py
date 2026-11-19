#!/usr/bin/env python3
"""Build one dynamic-batch GEM-X Hand ONNX from two numerically checked exports.

The reference exports must be identical except for the batch-dependent shape
initializers. This tool rewrites those shape initializers to computations from
Shape(img)[0], sharing the ORIGINAL batch-1 weights/operations at N=1 or N=2.

It does not quantize weights, duplicate two networks, or generate any training
data. It deliberately refuses unfamiliar graph differences. The resulting
ONNX is not a TensorRT speed/accuracy certification.

Example:
 python tools/quantize/dynamic_hand_batch.py build \
   --single model_comparison/hand_fp16.onnx \
   --double model_comparison/hand_batch2_fp16.onnx \
   --output out/hand_dynamic_fp16.onnx
 python tools/quantize/dynamic_hand_batch.py verify \
   --single model_comparison/hand_fp16.onnx \
   --double model_comparison/hand_batch2_fp16.onnx \
   --dynamic out/hand_dynamic_fp16.onnx
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT_NAMES = ("img", "bbx_xys", "K_fullimg", "ray_feats")
OUTPUT_NAMES = ("wrist_orient", "hand_pose", "identity_coeffs",
                "scale_params", "pred_cam", "pose_token",
                "kp2d_mu", "kp2d_presence")


def _named_inputs(model):
    return [x for x in model.graph.input
            if x.name not in {init.name for init in model.graph.initializer}]


def _graph_signature(single, double):
    """Require identical computation before changing any graph nodes."""
    if tuple(i.name for i in _named_inputs(single)) != INPUT_NAMES:
        raise ValueError("Not a GEM-X Hand ONNX with the expected four inputs")
    if tuple(i.name for i in _named_inputs(double)) != INPUT_NAMES:
        raise ValueError("Double ONNX has different input names")
    if tuple(o.name for o in single.graph.output) != OUTPUT_NAMES:
        raise ValueError("Unexpected Hand output names")
    if tuple(o.name for o in double.graph.output) != OUTPUT_NAMES:
        raise ValueError("Double ONNX has different output names")
    if len(single.graph.node) != len(double.graph.node):
        raise ValueError("Node counts differ; cannot safely derive a shared graph")
    differing_nodes = [
        (idx, a.op_type, a.name)
        for idx, (a, b) in enumerate(zip(single.graph.node, double.graph.node))
        if a.SerializeToString() != b.SerializeToString()
    ]
    if differing_nodes:
        raise ValueError(f"Graph nodes differ: {differing_nodes[:12]}")
    opset1 = [(x.domain, x.version) for x in single.opset_import]
    opset2 = [(x.domain, x.version) for x in double.opset_import]
    if opset1 != opset2:
        raise ValueError("ONNX opsets differ")
    if next((v for d, v in opset1 if d in ("", "ai.onnx")), 0) < 13:
        raise ValueError("ONNX opset >=13 required for dynamic Squeeze")
    for graph_name, values in (("input", zip(_named_inputs(single), _named_inputs(double))),
                               ("output", zip(single.graph.output, double.graph.output))):
        for a, b in values:
            at = a.type.tensor_type
            bt = b.type.tensor_type
            ashape = at.shape.dim
            bshape = bt.shape.dim
            if at.elem_type != bt.elem_type or len(ashape) != len(bshape) or not ashape:
                raise ValueError(f"Incompatible {graph_name} {a.name}")
            if ashape[0].dim_value != 1 or bshape[0].dim_value != 2:
                raise ValueError(f"Expected static batch 1/2 for {a.name}")
            for x, y in zip(ashape[1:], bshape[1:]):
                if x.SerializeToString() != y.SerializeToString():
                    raise ValueError(f"Nonbatch dimension changed: {a.name}")


def _sha(init):
    return hashlib.sha256(memoryview(init.raw_data)).digest()


def build(single_path, double_path, output):
    import onnx
    from onnx import helper, numpy_helper

    for path in (single_path, double_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or output.with_name(output.name + ".data").exists():
        raise FileExistsError("Refusing to overwrite an existing dynamic ONNX")
    if output.resolve() in (single_path.resolve(), double_path.resolve()):
        raise ValueError("The output must not replace the original ONNX")
    a = onnx.load(str(single_path))
    b = onnx.load(str(double_path))
    _graph_signature(a, b)

    weights_a = {x.name: x for x in a.graph.initializer}
    weights_b = {x.name: x for x in b.graph.initializer}
    if weights_a.keys() != weights_b.keys():
        raise ValueError("Models have different initializers")
    same = 0
    scaled = []
    expanded = []
    for name, v in weights_a.items():
        w = weights_b[name]
        if (v.data_type == w.data_type and tuple(v.dims) == tuple(w.dims)
                and _sha(v) == _sha(w)):
            same += 1
            continue
        if name.startswith("val_"):
            v_np = numpy_helper.to_array(v)
            w_np = numpy_helper.to_array(w)
            if (v.data_type not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64)
                    or v_np.shape != w_np.shape
                    or not np.array_equal(w_np.astype(np.int64),
                                          v_np.astype(np.int64) * 2)):
                raise ValueError(f"Unexpected changed shape scalar {name}")
            scaled.append(name)
        elif name.startswith("expand"):
            v_np = numpy_helper.to_array(v)
            w_np = numpy_helper.to_array(w)
            if (v_np.ndim < 1 or w_np.shape != (v_np.shape[0] * 2, *v_np.shape[1:])
                    or not np.array_equal(w_np, np.concatenate((v_np, v_np), axis=0))):
                raise ValueError(f"Unexpected changed expand initializer {name}")
            expanded.append(name)
        else:
            raise ValueError(f"Non-shape model weight differs: {name}")
    if not scaled or not expanded:
        raise ValueError("Batch-dependent shape constants not found; no dynamic rewrite")

    # Read the batch dimension from the actual input tensor, not from a
    # separately supplied number which could disagree with other inputs.
    # This provides ONE runtime batch parameter for ALL four input tensors.
    prefix = "__gemx_dynamic_batch__"
    existing_names = {x.name for x in a.graph.initializer} | {
        y for node in a.graph.node for y in (*node.input, *node.output)}
    if any(name.startswith(prefix) for name in existing_names):
        raise ValueError("Reserved dynamic-batch name already exists in the graph")
    init = [
        numpy_helper.from_array(np.asarray([0], dtype=np.int64),
                                name=prefix + "index"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64),
                                name=prefix + "squeeze_axis"),
    ]
    ops = [
        helper.make_node("Shape", ["img"], [prefix + "input_shape"],
                         name=prefix + "shape"),
        helper.make_node("Gather", [prefix + "input_shape", prefix + "index"],
                         [prefix + "batch_vec"], axis=0, name=prefix + "gather"),
        helper.make_node("Squeeze", [prefix + "batch_vec", prefix + "squeeze_axis"],
                         [prefix + "batch_scalar"], name=prefix + "squeeze"),
    ]

    for name in sorted(scaled):
        base = weights_a[name]
        if base.data_type == onnx.TensorProto.INT64:
            batch = prefix + "batch_scalar"
        elif base.data_type == onnx.TensorProto.INT32:
            batch = prefix + "batch_int32"
            if not any(x.output and x.output[0] == batch for x in ops):
                ops.append(helper.make_node("Cast", [prefix + "batch_scalar"], [batch],
                                            to=onnx.TensorProto.INT32,
                                            name=prefix + "cast_int32"))
        else:
            raise AssertionError(name)
        init.append(numpy_helper.from_array(numpy_helper.to_array(base),
                                             name=prefix + "base_" + name))
        ops.append(helper.make_node(
            "Mul", [batch, prefix + "base_" + name], [name],
            name=prefix + "mul_" + name))
    for name in sorted(expanded):
        base = weights_a[name]
        dims = np.asarray(base.dims[1:], dtype=np.int64)
        init.append(numpy_helper.from_array(dims, name=prefix + "tail_" + name))
        init.append(numpy_helper.from_array(numpy_helper.to_array(base),
                                             name=prefix + "base_" + name))
        shape_name = prefix + "shape_" + name
        ops.append(helper.make_node(
            "Concat", [prefix + "batch_vec", prefix + "tail_" + name],
            [shape_name], axis=0, name=prefix + "concat_" + name))
        ops.append(helper.make_node(
            "Expand", [prefix + "base_" + name, shape_name], [name],
            name=prefix + "expand_" + name))

    for entry in (*_named_inputs(a), *a.graph.output):
        dim = entry.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    # Existing intermediate value_info contains fixed single-batch sizes.
    # It is optional for runtime and must not conflict with the dynamic shapes.
    a.graph.ClearField("value_info")
    keep = [x for x in a.graph.initializer
            if x.name not in set(scaled) | set(expanded)]
    a.graph.ClearField("initializer")
    a.graph.initializer.extend(keep)
    a.graph.initializer.extend(init)
    old_nodes = list(a.graph.node)
    a.graph.ClearField("node")
    a.graph.node.extend(ops)
    a.graph.node.extend(old_nodes)

    del b
    gc.collect()
    # Avoid accidental publication of huge weights: only write to a local
    # file in Codespace. The generated ONNX external data stays alongside it.
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(a, str(output), save_as_external_data=True,
                    all_tensors_to_one_file=True, size_threshold=1024,
                    location=output.name + ".data")
    del a
    gc.collect()
    onnx.checker.check_model(str(output))
    report = {"source_single": str(single_path), "source_double": str(double_path),
              "dynamic_output": str(output), "unchanged_initializers": same,
              "dynamic_scaled_shapes": len(scaled),
              "dynamic_expanded_shapes": len(expanded),
              "status": "structurally checked; CPU inference not yet verified"}
    print(json.dumps(report, indent=2), flush=True)


def _feeds(inputs, count, rng):
    # Inputs are synthetic and produced locally: NEVER export private user frames.
    arrays = {}
    for info in inputs:
        if len(info.shape) == 0 or info.shape[0] not in (count, "batch", None):
            raise ValueError(f"Unexpected batch input {info.name}: {info.shape}")
        shape = (count, *[int(x) for x in info.shape[1:]])
        name = info.name
        if info.type == "tensor(float16)":
            dtype = np.float16
        elif info.type == "tensor(float)":
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported input type {name}: {info.type}")
        if name == "img" or name == "ray_feats":
            value = rng.normal(0, .3, size=shape).astype(dtype)
        elif name == "bbx_xys":
            value = np.tile([128., 128., 200.], (count, 1)).astype(dtype)
        elif name == "K_fullimg":
            value = np.tile(np.asarray([[256., 0., 128.], [0., 256., 128.],
                                        [0., 0., 1.]]), (count, 1, 1)).astype(dtype)
        else:
            raise ValueError(f"Unexpected input name {name}")
        arrays[name] = np.ascontiguousarray(value)
    return arrays


def verify(single, double, dynamic, atol, rtol):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    rng = np.random.default_rng(1428)
    result = {}
    # Only one >1GB ORT model session at a time; retain small outputs.
    for batch, reference_path in ((1, single), (2, double)):
        reference = ort.InferenceSession(str(reference_path),
                                         sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        feed = _feeds(reference.get_inputs(), batch, rng)
        ref_outputs = reference.run(None, feed)
        names = [x.name for x in reference.get_outputs()]
        del reference
        gc.collect()
        dyn = ort.InferenceSession(str(dynamic), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
        dyn_names = [x.name for x in dyn.get_outputs()]
        if dyn_names != names:
            raise ValueError("Output names differ")
        dyn_inputs = {x.name: x for x in dyn.get_inputs()}
        dyn_feed = {}
        for key, data in feed.items():
            if key not in dyn_inputs:
                raise ValueError(f"Missing dynamic input {key}")
            dyn_feed[key] = np.ascontiguousarray(
                data, dtype=np.float16 if dyn_inputs[key].type == "tensor(float16)"
                else np.float32)
        got = dyn.run(None, dyn_feed)
        del dyn
        gc.collect()
        stats = {}
        for name, a, b in zip(names, ref_outputs, got):
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"{name} N={batch}: shape or finite check failed")
            delta = np.abs(a - b)
            stats[name] = {"max_abs": float(delta.max(initial=0)),
                           "match": bool(np.allclose(a, b, atol=atol, rtol=rtol))}
        result[batch] = stats
        print(f"N={batch}: " + json.dumps(stats), flush=True)
        if not all(x["match"] for x in stats.values()):
            raise ValueError(f"Dynamic batch N={batch} output mismatch")
    print("PASS: same ONNX dynamic graph operates at N=1 and N=2",
          "with per-output numerical checks", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--single", type=Path, required=True)
        s.add_argument("--double", type=Path, required=True)
        if name == "build":
            s.add_argument("--output", type=Path, required=True)
        else:
            s.add_argument("--dynamic", type=Path, required=True)
            s.add_argument("--atol", type=float, default=0.01)
            s.add_argument("--rtol", type=float, default=0.01)
    args = ap.parse_args()
    if args.command == "build":
        build(args.single, args.double, args.output)
    else:
        verify(args.single, args.double, args.dynamic, args.atol, args.rtol)


if __name__ == "__main__":
    main()
