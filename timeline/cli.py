"""plan.json -> a minted .api.json, from a shell. The T0 rung, usable alone.

    python -m timeline.cli compile PLAN.json --graph OUT.api.json
    python -m timeline.cli price   PLAN.json [--port 8189]
    python -m timeline.cli new     CLIP.mp4 --frames 124 --ramp 73:2.0 \
                                   --out PLAN.json

It never queues anything: launching is queue_scene.py's job (the project's
only sanctioned launcher, which guards prefix collisions and refuses
partial graphs). This prints the exact command instead.

Run it from the pack directory, or with the pack on PYTHONPATH.
"""
import argparse
import json
import os
import sys

if __package__ in (None, ""):                    # `python timeline/cli.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import schema                                    # noqa: E402
from timeline.h3.compile import H3Backend                      # noqa: E402

# Site config: override via env for your install; defaults are the authoring box.
QUEUE = os.environ.get("H3_TIMELINE_QUEUE_CMD",
        "/mnt/work/ai/venvs/comfyui-cu132/bin/python "
        "/mnt/work/ai/apps/ComfyUI-ModelCatalog/benchmarks/scripts/"
         "queue_scene.py")


def _default_graph(plan_path):
    base = os.path.basename(plan_path).replace(".plan.json", "").replace(
        ".json", "")
    return os.path.join(os.environ.get("H3_TIMELINE_GRAPH_DIR",
                        "/mnt/work/ai/apps/ComfyUI-ModelCatalog/workflows"),
                        f"{base}.api.json")


def cmd_new(a):
    plan = schema.new_plan(a.clip, frames=a.frames, fps=a.fps,
                           width=a.width, height=a.height,
                           proposed_by="timeline.cli",
                           settings={"prompt": a.prompt, "seed": a.seed,
                                     "steps": a.steps,
                                     "regen_strength": a.regen_strength,
                                     "output_prefix": a.output_prefix})
    env = [[0, 1.0]]
    for spec in a.ramp:
        f, r = spec.split(":")
        env.append([int(f), float(r)])
    if env[-1][0] != a.frames - 1:
        env.append([a.frames - 1, env[-1][1]])
    plan["lanes"].append(schema.generation_density_lane(
        env, ceiling=a.ceiling, proposer="timeline.cli"))
    schema.validate_or_raise(plan)
    print("wrote", schema.save(plan, a.out))
    print(json.dumps(plan["lanes"][0]["envelope"]))


def cmd_compile(a):
    plan = schema.load(a.plan)
    problems = schema.validate(plan)
    if problems:
        sys.exit("INVALID PLAN: " + "; ".join(problems))
    graph = a.graph or _default_graph(a.plan)
    res = H3Backend().compile(plan, {"graph_path": graph, "port": a.port,
                                     "recorder_path": a.recorder})
    schema.save(plan, a.plan)
    print(res.report)
    print("minted:", graph)
    print("plan updated (compiled cache):", a.plan)
    print("launch:", f"{QUEUE} {graph} --tag timeline --port {a.port}")


def cmd_price(a):
    plan = schema.load(a.plan)
    est = H3Backend().estimate(plan, {"port": a.port, "recorder_path": a.recorder})
    print(est)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="timeline.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new", help="write a plan with one density ramp")
    n.add_argument("clip")
    n.add_argument("--frames", type=int, required=True)
    n.add_argument("--fps", type=float, default=24.0)
    n.add_argument("--width", type=int, default=1152)
    n.add_argument("--height", type=int, default=640)
    n.add_argument("--ramp", action="append", default=[],
                   metavar="FRAME:DENSITY",
                   help="density envelope control point (generated frames "
                        "per world frame; 1.0 = plain)")
    n.add_argument("--ceiling", type=float, default=4.0)
    n.add_argument("--prompt", default="")
    n.add_argument("--seed", type=int, default=20260817)
    n.add_argument("--steps", type=int, default=25)
    n.add_argument("--regen-strength", dest="regen_strength", type=float,
                   default=0.45)
    n.add_argument("--output-prefix", dest="output_prefix",
                   default="video/timeline")
    n.add_argument("--out", required=True)
    n.set_defaults(func=cmd_new)

    c = sub.add_parser("compile", help="plan -> minted .api.json")
    c.add_argument("plan")
    c.add_argument("--graph", default=None)
    c.add_argument("--port", type=int, default=8189)
    c.add_argument("--recorder", default=None)
    c.set_defaults(func=cmd_compile)

    p = sub.add_parser("price", help="price a plan without minting")
    p.add_argument("plan")
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--recorder", default=None)
    p.set_defaults(func=cmd_price)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
