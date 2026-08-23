#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow to the /prompt API format.

The frontend normally does this in the browser. The rule it follows, and that
this reproduces: widgets_values maps POSITIONALLY onto the node's widget-type
inputs in INPUT_TYPES order (required then optional), where "widget-type" means
anything that isn't a link-only socket. An input carrying control_after_generate
eats one extra widgets_values slot for the control mode itself. Linked inputs
then override whatever the widget slot held.

Two more frontend behaviours live here because 0.33 workflows need them:

* SUBGRAPHS. workflow["definitions"]["subgraphs"] holds reusable node groups;
  a node whose "type" is one of their uuids is an instance. They nest. We
  flatten them into plain nodes before conversion — inner links to the virtual
  input node (id -10) resolve to whatever drives the instance's matching input
  slot (a parent link, or a promoted widget literal), and the parent's readers
  of an instance output slot resolve to the inner node feeding the virtual
  output node (id -20).
* MUTE (mode 2) / BYPASS (mode 4). Bypass is pass-through: a reader of a
  bypassed output gets rewired to that node's first same-typed input, trying
  the same slot index first, exactly as the frontend does. Mute just deletes.
  Either way a link can go dead; the reader's input is then dropped, and any
  node left missing a REQUIRED input is pruned (and so on transitively), so a
  muted branch nobody needs cannot block the whole conversion.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

# run_bench.py already honors COMFY_API; this must match it, or a node that
# exists only on one instance (e.g. a custom node after a single-instance
# restart) fails object_info lookup against the wrong server.
API = os.environ.get("COMFY_API", "http://127.0.0.1:8188")

# Sockets that can only be fed by a link — never consume a widgets_values slot.
LINK_ONLY = {
    "MODEL", "CLIP", "VAE", "IMAGE", "MASK", "AUDIO", "LATENT", "CONDITIONING",
    "SAMPLER", "SIGMAS", "GUIDER", "NOISE", "VIDEO", "CLIP_VISION",
    "CONTROL_NET", "STYLE_MODEL", "COMFY_AUTOGROW_V3",
}
# Frontend-only nodes: they have no server schema and never reach /prompt.
# The rgthree fast-groups panels toggle other nodes' modes and are pure UI.
SKIP_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode",
              "Label (rgthree)", "Fast Groups Muter (rgthree)",
              "Fast Groups Bypasser (rgthree)"}

_oi_cache = {}


def object_info(node_type):
    if node_type not in _oi_cache:
        # node types can contain spaces ("Any Switch (rgthree)") — quote or
        # http.client rejects the URL outright.
        url = f"{API}/object_info/{urllib.parse.quote(node_type)}"
        with urllib.request.urlopen(url) as r:
            schema = json.load(r).get(node_type)
        if schema is None:                   # node not installed on this server
            print(f"ui2api: no schema for {node_type!r}; widgets left unmapped",
                  file=sys.stderr)
        _oi_cache[node_type] = schema or {}
    return _oi_cache[node_type]


def widget_slots(node_type):
    """Ordered [(name, eats_extra_slot)] for the node's widget-type inputs."""
    inp = object_info(node_type).get("input") or {}
    out = []
    for sec in ("required", "optional"):
        for name, spec in (inp.get(sec) or {}).items():
            t = spec[0]
            if not isinstance(t, list) and t in LINK_ONLY:
                continue
            meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            out.append((name, bool(meta.get("control_after_generate"))))
    return out


def required_names(node_type):
    return set(((object_info(node_type).get("input") or {}).get("required") or {}).keys())


def _link_map(links):
    """link id -> (origin_id, origin_slot). Top level stores tuples, subgraph
    definitions store dicts; same information either way."""
    m = {}
    for l in links or []:
        if isinstance(l, dict):
            m[l["id"]] = (l["origin_id"], l["origin_slot"])
        else:
            m[l[0]] = (l[1], l[2])
    return m


def _widget_inputs(node):
    """Instance sockets that a promoted widget feeds, in widgets_values order."""
    return [s for s in (node.get("inputs") or []) if s.get("widget")]


def flatten(wf):
    """Inline every subgraph instance, recursively, into plain nodes.

    Emits UI-format nodes with namespaced string ids ("882:113" = node 113 of
    the instance at 882, the frontend's own spelling) and a fresh link table. A boundary crossing that lands
    on a promoted widget rather than a link becomes a literal in "_literals",
    which convert() applies on top of the inner node's own widgets_values.
    """
    defs = {s["id"]: s for s in ((wf.get("definitions") or {}).get("subgraphs") or [])}
    if not defs:                              # untouched: keeps old output byte-identical
        return wf

    nodes, links = [], []                     # accumulated flat graph
    counter = [0]
    pending = []                              # scopes whose nodes still need emitting

    def new_link(origin):                     # origin = (id, slot)
        counter[0] += 1
        links.append([counter[0], origin[0], origin[1], 0, 0, ""])
        return counter[0]

    def scope(snodes, slinks, ns, inbound):
        """Expand one graph level. inbound(slot) resolves the virtual input
        node (-10); returns outbound(slot) resolving the virtual output (-20).
        Resolutions are ('node', id, slot) or ('value', v) or None."""
        by_id = {n["id"]: n for n in snodes}
        lm = _link_map(slinks)
        expanded = {}                         # instance id -> outbound resolver
        emitted = set()

        def resolve(origin_id, origin_slot, guard=0):
            if guard > 32:
                return None
            if origin_id == -10:
                return inbound(origin_slot)
            n = by_id.get(origin_id)
            if n is None:
                return None
            if n.get("type") == "Reroute":    # virtual: read straight through
                up = (n.get("inputs") or [{}])[0].get("link")
                return resolve(*lm[up], guard=guard + 1) if up in lm else None
            if n.get("type") in defs:
                return instance(n)(origin_slot)
            emit(n)
            return ("node", f"{ns}{origin_id}", origin_slot)

        def socket_source(n, socket):
            lid = socket.get("link")
            return resolve(*lm[lid]) if lid in lm else None

        def instance(n):
            if n["id"] in expanded:
                return expanded[n["id"]]
            sub = defs[n["type"]]
            wv = n.get("widgets_values") or []
            promoted = {s["name"]: i for i, s in enumerate(_widget_inputs(n))}

            def child_inbound(slot):
                socks = n.get("inputs") or []
                if slot >= len(socks):
                    return None
                s = socks[slot]
                if s.get("link") is not None:
                    return socket_source(n, s)
                i = promoted.get(s["name"])
                if i is not None and i < len(wv):
                    return ("value", wv[i])
                return None                   # unfed: inner node keeps its own widget

            if n.get("mode") == 2:            # muted: the interior never runs
                out = lambda slot: None
            elif n.get("mode") == 4:          # bypassed: pass through the box
                def out(slot, n=n, sub=sub):
                    outs = sub.get("outputs") or []
                    want = outs[slot].get("type") if slot < len(outs) else None
                    socks = n.get("inputs") or []
                    order = [slot] + [i for i in range(len(socks)) if i != slot]
                    for i in order:
                        if i < len(socks) and socks[i].get("type") == want:
                            return child_inbound(i)
                    return None
            else:
                out = scope(sub.get("nodes") or [], sub.get("links") or [],
                            f"{ns}{n['id']}:", child_inbound)
            expanded[n["id"]] = out
            return out

        def emit(n):
            if n["id"] in emitted:
                return
            emitted.add(n["id"])
            copy = dict(n)
            copy["id"] = f"{ns}{n['id']}"
            literals, socks = {}, []
            for s in n.get("inputs") or []:
                s = dict(s)
                src = socket_source(n, s)
                if src and src[0] == "node":
                    s["link"] = new_link((src[1], src[2]))
                elif src and src[0] == "value":
                    literals[s["name"]] = src[1]
                    s["link"] = None
                elif s.get("link") is not None:
                    s["link"] = None          # dead across the boundary
                socks.append(s)
            copy["inputs"] = socks
            if literals:
                copy["_literals"] = literals
            nodes.append(copy)

        def emit_all():                       # everything runs, reachable or not
            for n in snodes:
                instance(n) if n.get("type") in defs else emit(n)

        # deferred: two instances can each read one of the other's outputs
        # (no cycle, different slots), and emitting eagerly here would ask for
        # an output of a scope that is still being built.
        pending.append(emit_all)

        def outbound(slot):
            for l in (slinks or []):
                tid = l["target_id"] if isinstance(l, dict) else l[3]
                tslot = l["target_slot"] if isinstance(l, dict) else l[4]
                if tid == -20 and tslot == slot:
                    oid = l["origin_id"] if isinstance(l, dict) else l[1]
                    oslot = l["origin_slot"] if isinstance(l, dict) else l[2]
                    return resolve(oid, oslot)
            return None
        return outbound

    scope(wf["nodes"], wf.get("links") or [], "", lambda slot: None)
    while pending:
        pending.pop(0)()
    out = dict(wf)
    out["nodes"], out["links"] = nodes, links
    out.pop("definitions", None)
    return out


def convert(wf):
    wf = flatten(wf)
    by_id = {n["id"]: n for n in wf["nodes"]}
    # link id -> (origin_node_id, origin_slot)
    links = {l[0]: (l[1], l[2]) for l in wf.get("links", [])}

    def source(origin_id, origin_slot, want, guard=0):
        """Walk virtual (Reroute) and bypassed (mode 4) nodes to a real origin.
        Bypass looks for an input of the requested type, same slot index first,
        which is the frontend's rule; no match means the link is dead."""
        while guard < 32:
            n = by_id.get(origin_id)
            if n is None:
                return None
            mode, t = n.get("mode"), n.get("type")
            if mode == 2:                     # muted: never executes
                return None
            if t != "Reroute" and mode != 4:
                return (origin_id, origin_slot)
            socks = n.get("inputs") or []
            if t == "Reroute":
                cand = [0]
            else:
                cand = [origin_slot] + [i for i in range(len(socks)) if i != origin_slot]
            for i in cand:
                if i >= len(socks):
                    continue
                st = socks[i].get("type")
                if t != "Reroute" and want not in (None, "*", st) and st != "*":
                    continue
                lid = socks[i].get("link")
                if lid is None or lid not in links:
                    continue
                origin_id, origin_slot = links[lid]
                break
            else:
                return None
            guard += 1
        return None

    prompt, dead = {}, {}
    for n in wf["nodes"]:
        t = n.get("type")
        if not t or t in SKIP_TYPES:
            continue
        if n.get("mode") in (2, 4):          # muted / bypassed: never executes
            continue
        inputs = {}

        # 1. widgets, positionally
        wv = n.get("widgets_values") or []
        if isinstance(wv, dict):             # some nodes store a dict (VHS etc.)
            inputs.update(wv)
        else:
            i = 0
            for name, eats_extra in widget_slots(t):
                if i >= len(wv):
                    break
                inputs[name] = wv[i]
                i += 1 + (1 if eats_extra else 0)

        # 2. literals promoted across a subgraph boundary
        inputs.update(n.get("_literals") or {})

        # 3. links override
        for socket in n.get("inputs") or []:
            lid = socket.get("link")
            if lid is None or lid not in links:
                continue
            origin_id, origin_slot = links[lid]
            if origin_id not in by_id:
                continue
            src = source(origin_id, origin_slot, socket.get("type"))
            if src is None:                  # muted/bypassed away: no source left
                dead.setdefault(str(n["id"]), set()).add(socket["name"])
                inputs.pop(socket["name"], None)
                continue
            inputs[socket["name"]] = [str(src[0]), src[1]]

        prompt[str(n["id"])] = {"class_type": t, "inputs": inputs,
                                "_meta": {"title": n.get("title") or t}}

    # 4. a dead link that fed a REQUIRED input means the node cannot run; drop
    #    it, and whatever read it loses its source in turn. Only reached when
    #    something actually died, so clean graphs come out untouched.
    pruned = set()
    while dead:
        drop = set()
        for nid, node in prompt.items():
            req = required_names(node["class_type"])
            miss = set(dead.get(nid, ()))
            for name, v in node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and v[0] in pruned:
                    miss.add(name)
            if miss & req:
                drop.add(nid)
        if not drop:
            break
        pruned |= drop
        for nid in drop:
            del prompt[nid]
            dead.pop(nid, None)
    for node in prompt.values():             # optional inputs just lose the ref
        for name, v in list(node["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and v[0] in pruned:
                del node["inputs"][name]
    return prompt


if __name__ == "__main__":
    wf = json.load(open(sys.argv[1]))
    api = convert(wf)
    json.dump(api, open(sys.argv[2], "w"), indent=2)
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(api)} nodes")
