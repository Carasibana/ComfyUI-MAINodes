# Persistent Generative World Architecture
## H3 Ref2VA + Persistent Gaussian World Memory + Agentic Composition

**Status:** Working design / architecture thesis  
**Primary goal:** Build a persistent generative video world whose visual and spatial state survives across shots, scenes, camera changes, and long time gaps without requiring the video model itself to remember everything.

---

## 1. Executive Summary

The core idea is to stop treating long-form video coherence as something a single video model must solve internally.

Instead, we externalize persistence into a **world memory layer** and let H3 act as the high-capability multimodal generator that reads from that memory, creates new observations of the world, and writes trustworthy changes back into it.

The preferred world representation is a **persistent Gaussian-splat map** that behaves more like a visual save-game or minimap than a perfect photogrammetric reconstruction.

The map does not need to look cinematic by itself. It only needs to preserve enough information to tell H3:

- where things are,
- what the room roughly looks like,
- what changed,
- what did not change,
- what is currently visible,
- what remains unobserved,
- and how a requested camera should move through the space.

The fundamental persistence rule is:

> **A new shot only has authority over regions it actually observes with sufficient confidence. Unobserved regions remain untouched.**

This allows a room to evolve incrementally.

If a chair moves in one shot, we update that chair.

If a glass disappears, we remove that glass only when the old location was clearly observed.

If the far side of the room was not visible, the old state survives exactly as it was.

The video model therefore does not need infinite memory. It receives **retrieval-augmented visual memory** from the persistence layer whenever it needs to revisit the world.

---

# 2. Why This Architecture

Modern generative video models are becoming powerful enough that many problems previously assumed to require monolithic end-to-end model improvements can instead be solved through:

- agentic composition,
- multimodal retrieval,
- reference selection,
- persistent external state,
- targeted inpainting/editing,
- LoRAs,
- occasional full fine-tunes,
- deterministic 3D tools,
- reconstruction tools,
- and iterative validation.

The architectural thesis is:

> **We may already have enough raw generative capability. The remaining gap is increasingly orchestration, persistence, control, and error correction.**

Rather than waiting for a hypothetical future model to perfectly understand a feature-length world's state, we can build that memory ourselves.

H3 is particularly attractive because its reference-to-video behavior makes the world memory useful even when the memory itself is visually crude.

The persistence layer does not have to generate the final frame.

It has to generate a strong enough **reference observation** for H3 to understand the intended world.

---

# 3. H3 Is the Center, Not Just the Renderer

An earlier version of the architecture placed H3 too far downstream, as though it were primarily a rendering endpoint.

That is not the preferred mental model.

H3 Ref2VA should be treated as the **multimodal generative core**.

It can combine different reference roles such as:

- character appearance,
- object appearance,
- environment appearance,
- style,
- action,
- pose,
- camera behavior,
- motion,
- temporal structure,
- prior footage,
- continuation context,
- and audio.

This means a shot does not need a single perfect reference clip.

The agent can construct a **shot-specific reference package** from many different sources.

Example:

```text
IMAGE 1  -> protagonist identity closeup
IMAGE 2  -> protagonist full-body costume reference
IMAGE 3  -> canonical kitchen appearance
IMAGE 4  -> refrigerator detail
IMAGE 5  -> secondary character identity
IMAGE 6  -> lighting/look reference

VIDEO 1  -> tail of previous shot
VIDEO 2  -> crude engine/splat camera playblast
VIDEO 3  -> hand/object interaction reference

AUDIO 1  -> protagonist voice
AUDIO 2  -> secondary character voice
AUDIO 3  -> room ambience
```

H3 can then synthesize these into a coherent shot.

The important shift is:

> **The persistence system does not need to contain every piece of information in final-quality form. It only needs to supply the correct constraints and references.**

---

# 4. The "Minimap" Mental Model

The persistence layer should not be thought of as a fully accurate digital twin.

A better analogy is a **generative minimap**.

It contains enough persistent information to keep the world coherent.

For a room, that might include:

```text
Kitchen
├── walls
├── windows
├── doors
├── countertop
├── refrigerator
├── table
│   ├── glass_04
│   ├── plate_07
│   └── keys_01
├── chair_01
├── chair_02
├── lighting anchors
└── appearance memory
```

The Gaussian representation gives this map something a pure symbolic world model does not have:

> **visual memory attached to spatial memory.**

The system remembers not only that a countertop exists, but approximately how that countertop appeared from previously observed views.

This is valuable because H3 can consume rendered observations of that state directly.

---

# 5. Persistent Gaussian World State

The preferred persistence behavior is incremental.

Start with a canonical room splat.

Each trustworthy new shot becomes a possible update.

```text
                 CANONICAL ROOM STATE
                 Persistent Gaussian Map
                          |
             +------------+------------+
             |                         |
       Visible in shot             Not visible
             |                         |
             v                         |
      New observation                  |
             |                         |
      classify regions                 |
             |                         |
      +------+------+                  |
      |             |                  |
    same          changed           untouched
      |             |                  |
    keep        replace/update         keep
      |             |                  |
      +------+------+------------------+
             |
             v
       NEW CANONICAL STATE
```

The key rules are:

1. **Observed and unchanged**
   - retain the canonical state.

2. **Observed and changed**
   - update only the changed region.

3. **Observed but uncertain**
   - retain old state unless confidence is sufficient.

4. **Unobserved**
   - never overwrite.

This prevents one generated shot from accidentally destroying parts of the world that were not even visible.

---

# 6. World Updates Are Transactions

Every accepted shot can be viewed as a transaction against world state.

Suppose the current kitchen contains:

```text
table
glass_04
chair_02
east_wall
west_wall
refrigerator
countertop
```

A generated shot observes:

```text
table
chair_02
east_wall
glass_04 location
```

The analysis might produce:

```text
table        -> observed / unchanged
chair_02     -> observed / moved
east_wall    -> observed / unchanged
glass_04     -> observed / removed
west_wall    -> unobserved
refrigerator -> unobserved
countertop   -> unobserved
```

Commit:

```text
table        -> keep
chair_02     -> update transform or local geometry
east_wall    -> keep
glass_04     -> remove
west_wall    -> preserve old state
refrigerator -> preserve old state
countertop   -> preserve old state
```

That is the desired persistence behavior.

---

# 7. Preserve History Instead of Destructive Overwrite

The system should ideally maintain state history.

Example:

```text
Shot 410 -> painting hanging on wall
Shot 411 -> painting falls
Shot 430 -> painting is rehung
```

Instead of discarding the original visual state, the world can maintain deltas:

```text
room_001
├── base_splat
├── delta_0001
├── delta_0002
└── delta_0003
```

Or object history:

```text
painting_07
├── state_01_hanging
├── state_02_floor
└── state_03_rehung
```

Benefits:

- rollback,
- alternate timelines,
- flashbacks,
- continuity debugging,
- reconstruction of earlier shots,
- branchable world state,
- cheaper storage than full checkpoints,
- and reusing prior object states instead of relearning them.

This also makes the architecture useful beyond linear filmmaking.

It can support persistent games, simulations, virtual worlds, and interactive narrative systems.

---

# 8. Local Coordinate Systems

Objects and regions should avoid permanently baking all Gaussian primitives into a single global coordinate frame.

Prefer a hierarchy:

```text
world transform
    *
region transform
    *
object transform
    *
Gaussian local position
```

Example:

```text
WORLD
├── room_kitchen
│   ├── static_shell
│   ├── table_01
│   ├── chair_01
│   ├── chair_02
│   └── glass_04
└── hallway
```

If `chair_02` moves, the system may only need to update:

```text
chair_02.transform
```

instead of modifying thousands of Gaussian primitives.

Only when an object's geometry actually changes should the system perform local Gaussian reconstruction.

This gives a useful hierarchy of corrections:

```text
KEEP
RIGID_REALIGN
SIM3_REALIGN
RECOLOR
LOCAL_OPTIMIZE
PRUNE_AND_RESPLAT
FULL_REGION_RECONSTRUCT
```

An agent can decide which operation is appropriate.

---

# 9. QuerySplat's Role

QuerySplat is potentially very useful, but it should not itself be the persistent memory.

Its best role is:

> **Lift a newly generated H3 observation back into candidate 3D.**

Conceptually:

```text
Persistent World Splat
        |
        v
Render requested camera/reference
        |
        v
H3 Ref2VA
        |
        v
Beautiful new shot
        |
        v
Select useful frames
        |
        v
QuerySplat
        |
        v
Candidate 3D observation
        |
        v
Register against persistent world
        |
        v
Detect trustworthy changes
        |
        v
Local commit
```

QuerySplat turns generated frames into something that can be compared spatially against the current world state.

Useful outputs may include:

- camera estimates,
- depth,
- depth confidence,
- point clouds,
- Gaussian primitives,
- and a coherent local coordinate interpretation.

Those are all useful for deciding whether a region is allowed to modify canonical memory.

---

# 10. QuerySplat Must Not Directly Replace the World

A newly reconstructed splat should be treated as an **observation**, not as truth.

Do not do:

```text
persistent_world = QuerySplat(new_frames)
```

Instead:

```text
candidate = QuerySplat(new_frames)

candidate
    |
    v
register to canonical map
    |
    v
visibility analysis
    |
    v
confidence analysis
    |
    v
semantic correspondence
    |
    v
geometric difference analysis
    |
    v
local change mask
    |
    v
commit approved regions only
```

Two independently generated splats may represent the same physical surface using different Gaussian primitives.

Therefore the merge logic must operate on spatial/semantic evidence rather than assuming Gaussian-to-Gaussian identity.

---

# 11. Related Tool Roles

The emerging stack can be separated by responsibility.

## H3 Ref2VA

**Role:** generative world observation and cinematic synthesis.

Responsibilities:

- combine visual references,
- preserve identities,
- follow motion references,
- synthesize camera behavior,
- create final-quality shots,
- edit bad regions,
- perform continuation,
- and create plausible missing observations.

---

## QuerySplat

**Role:** observation lifting.

Responsibilities:

- take H3-generated frames,
- infer coherent local 3D,
- estimate cameras/depth,
- create a candidate Gaussian representation,
- and provide data for world-state comparison.

---

## Persistent Gaussian Map

**Role:** long-term visual/spatial memory.

Responsibilities:

- retain rooms and objects,
- preserve unseen geometry,
- track visual state,
- store confidence,
- store current and historical state,
- generate rough views from arbitrary cameras.

---

## Local Update / Continual-GS Techniques

Concepts from systems such as CL-Splats and related continual reconstruction work are relevant.

**Role:** safe mutation.

Responsibilities:

- detect changed spatial regions,
- freeze untouched regions,
- update only affected Gaussians,
- maintain compact deltas,
- and avoid catastrophic map replacement.

---

## Change Detection

Gaussian-space or hybrid 2D/3D change detection can act as the gatekeeper.

**Role:** determine what actually changed.

Possible classifications:

```text
UNCHANGED
RIGID_MOVEMENT
GEOMETRY_CHANGE
APPEARANCE_CHANGE
REMOVED
NEW_OBJECT
UNCERTAIN
UNOBSERVED
```

---

## Game Engine

**Role:** optional deterministic topology/physics oracle.

A game engine remains useful, but it does not need to render the final movie.

It can provide:

- collision,
- object transforms,
- door state,
- skeletal pose,
- camera trajectories,
- rough lighting direction,
- room topology,
- physics,
- object identity,
- and deterministic blocking.

The guiding principle is:

> **Use the game engine to tell H3 what happened, not necessarily what the movie should look like.**

An ugly playblast can still be an excellent motion/spatial reference.

---

# 12. 3D -> H3 -> 3D Loop

One of the most important architectural loops is:

```text
              Persistent 3D World
                      |
                      v
              Render reference
                      |
                      v
                     H3
              repair / extend /
              animate / render
                      |
                      v
              Generated video
                      |
                      v
                 QuerySplat
                      |
                      v
             Candidate new 3D
                      |
                      v
               Validate/merge
                      |
                      v
              Persistent 3D World
```

Or more simply:

> **3D -> H3 -> 3D -> H3 -> 3D**

This lets the world evolve over time.

H3 can imagine a new observation.

QuerySplat or similar tools can lift that observation back into spatial memory.

The persistence layer then decides whether the new information is trustworthy enough to become canonical.

---

# 13. Expanding Into Previously Unseen Space

This architecture becomes especially interesting when the camera enters a region that has never been modeled.

Suppose the world contains:

```text
kitchen
hallway
living room
unknown room
```

The system can:

1. render the known hallway and doorway,
2. request a camera trajectory into the unknown area,
3. give H3 canonical style/environment references,
4. ask H3 to generate the continuation,
5. reconstruct useful frames into a candidate 3D region,
6. align that region to the known doorway/hallway,
7. validate it,
8. commit it as a new part of the world.

Conceptually:

```text
KNOWN WORLD
    |
    v
reference camera trajectory
    |
    v
H3 generates plausible unseen space
    |
    v
QuerySplat
    |
    v
new candidate geometry
    |
    v
align to known anchors
    |
    v
commit
    |
    v
WORLD EXPANDS
```

Once committed, that room is no longer pure hallucination.

It has become persistent world memory.

Future shots can retrieve and render it again.

---

# 14. Sparse Skeleton, Not Perfect Reconstruction

The Gaussian world does **not** need to be a perfect photorealistic scene.

This is an important design choice.

The goal is not necessarily:

> render the persistent splat directly as the final image.

The goal is:

> render enough spatial truth for H3 to reconstruct the intended world.

The persistent world mainly needs:

- surfaces,
- relative layout,
- rough material/color cues,
- object placement,
- depth,
- occlusion,
- camera parallax,
- persistent clutter,
- visual anchors.

If the far side of a couch contains ugly sparse Gaussians, that may be acceptable.

H3 can clean up the appearance.

This lowers the reconstruction burden dramatically.

---

# 15. Confidence Is a First-Class Property

Every persistent region should carry confidence.

Possible fields:

```text
position
rotation
scale
appearance
semantic_id
object_id

last_observed_shot
observation_count

geometry_confidence
appearance_confidence
identity_confidence
registration_confidence

state:
  canonical
  changed
  uncertain
  historical
```

A low-confidence region should not be allowed to overwrite a high-confidence canonical region.

This is especially important with generated video because visual plausibility does not imply spatial truth.

The commit system should distinguish:

> "H3 generated something convincing"

from:

> "we have enough evidence that the canonical world really changed."

---

# 16. Persistent State vs References vs LoRAs

Different forms of memory should have different jobs.

## World Map = Episodic / Mutable Facts

Examples:

- the glass currently sits on the table,
- the chair is rotated 20 degrees,
- the door is open,
- the character left the room,
- the painting fell,
- the refrigerator contains a cake.

These facts should not require fine-tuning.

---

## Reference Library = Canonical Visual Memory

Examples:

- exact face reference,
- exact costume,
- canonical room look,
- prop details,
- lighting style,
- known camera treatment,
- prior shots from useful angles.

The agent retrieves these when preparing a shot.

---

## LoRAs = Persistent Learned Priors

LoRAs are useful when the base model repeatedly fails to honor references reliably.

Examples:

- character identity,
- recurring creature design,
- studio aesthetic,
- specific cinematography grammar,
- hard-to-preserve wardrobe,
- unusual recurring objects,
- production-specific motion language.

The principle is:

> **Do not encode transient world facts into a LoRA.**

A moved chair belongs in the map.

A stubborn character identity may belong in a LoRA.

---

## Full Fine-Tunes = Behavioral / Domain Priors

Full fine-tuning may eventually be appropriate for:

- production-specific reference following,
- particular cinematographic behavior,
- better persistence responses,
- improved handling of our rendered conditioning style,
- better interaction with synthetic engine/splat references,
- or specific animation domains.

But full fine-tuning is not the first requirement.

The external architecture should work independently.

---

# 17. Agentic Reference Composition

The world may eventually contain tens of thousands of possible reference assets.

H3 should not receive all of them.

An agent should build a **shot-specific multimodal context pack**.

Inputs to the agent:

```text
script state
scene state
camera request
characters present
objects visible
current world map
previous shot
continuity requirements
style requirements
motion requirements
audio requirements
```

Output:

```text
selected image references
selected video references
selected audio references
persistent-world render
engine playblast if useful
previous-shot tail
LoRA set
prompt / structured instructions
```

The agent is effectively performing:

> **multimodal RAG for video generation.**

Instead of retrieving text passages, it retrieves:

- images,
- videos,
- audio,
- geometry,
- world state,
- object state,
- previous shots,
- motion clips,
- and learned adapters.

---

# 18. Generation and Validation Loop

The production loop can be:

```text
                     STORY / TASK
                          |
                          v
                    Director Agent
                          |
                          v
                   Shot Requirements
                          |
                          v
              Retrieve Persistent State
                          |
                          v
               Build Reference Package
                          |
                          v
                         H3
                          |
                          v
                  Generated Candidate
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Visual Evaluator        Spatial Evaluator
              |                       |
              +-----------+-----------+
                          |
                          v
                 Trustworthy result?
                     /          \
                   yes           no
                    |             |
                    v             v
            Commit observations   edit/inpaint/
              to world state      regenerate
                    |
                    v
             Updated world memory
```

The world map therefore becomes both:

- an input to generation,
- and a recipient of validated output.

---

# 19. The Commit Algorithm

The commit algorithm may be the most important technical component in the whole architecture.

Its responsibility:

> Given the previous canonical world and a newly generated H3 clip, determine exactly which spatial regions are allowed to modify persistent state.

A possible pipeline:

```text
H3 clip
  |
  v
frame selection
  |
  v
QuerySplat / depth / camera inference
  |
  v
candidate local 3D
  |
  v
register candidate to canonical world
  |
  v
visibility determination
  |
  v
semantic association
  |
  v
geometry + appearance comparison
  |
  v
confidence calculation
  |
  v
change classification
  |
  v
approved mutation mask
  |
  v
transactional world update
```

Important rules:

### Rule 1: Unobserved Means Immutable

If the new shot does not see a region, the region cannot be modified.

### Rule 2: Low Confidence Cannot Destroy High Confidence

A weak reconstruction cannot overwrite a stable canonical surface.

### Rule 3: Rigid Movement Should Prefer Transform Updates

Do not reconstruct an object if its identity and geometry are still valid.

### Rule 4: Appearance and Geometry Are Different

A lighting/color shift should not automatically be interpreted as geometry change.

### Rule 5: Deletion Requires Evidence

An object disappears from the world only when its former location is sufficiently observed and proven empty.

### Rule 6: New Geometry Begins as Provisional

Newly hallucinated regions may require repeated confirmation before becoming high-confidence canonical state.

### Rule 7: Keep History

Accepted updates should preferably be stored as deltas or versioned states.

---

# 20. Example: Character Moves a Chair and Takes a Glass

Initial state:

```text
Kitchen:
  chair_02 = at table, rotation 0 degrees
  glass_04 = on table, 35% full
```

Shot action:

```text
Sarah enters.
Sarah rotates chair_02.
Sarah picks up glass_04.
Camera follows Sarah out.
```

After generation:

```text
QuerySplat / reconstruction
        |
        v
compare to canonical kitchen
```

Detected:

```text
chair_02 -> same object, new rigid transform
glass_04 -> removed from table
table -> unchanged
east wall -> unchanged
refrigerator -> unobserved
countertop -> unobserved
```

Commit:

```text
chair_02.transform = new transform

glass_04.location = Sarah/right_hand
or later:
glass_04.location = unknown/carried

table = unchanged
east wall = unchanged

refrigerator = untouched
countertop = untouched
```

No full room reconstruction occurs.

---

# 21. Example: Geometry Actually Changes

Suppose a table is broken.

This is not a rigid transform.

The agent may choose:

```text
LOCAL_OPTIMIZE
```

or:

```text
PRUNE_AND_RESPLAT
```

Pipeline:

```text
detect changed table region
        |
        v
freeze rest of room
        |
        v
remove invalid table Gaussians
        |
        v
reconstruct broken geometry
        |
        v
blend boundary
        |
        v
commit table delta
```

Everything else remains stable.

---

# 22. Example: H3 Drifts on Character Identity

Suppose H3 repeatedly changes a character's face at certain angles even with good reference retrieval.

Do not change the world map.

Instead:

```text
collect successful identity observations
        |
        v
train/refine character identity LoRA
        |
        v
evaluate against reference-only baseline
        |
        v
use LoRA when confidence improvement is real
```

This separates:

```text
WORLD FACT
"Sarah is standing by the refrigerator."
```

from:

```text
MODEL PRIOR
"This is what Sarah looks like from every angle."
```

---

# 23. Why Gaussian Splats Instead of Only a Game Engine

A game engine is extremely useful for exact state.

But a pure engine representation has a weakness:

> it stores geometry/state well, but it does not naturally retain the generative model's actual visual memory of the environment.

Gaussian splats give us:

- view-dependent appearance,
- captured texture,
- clutter,
- lighting cues,
- imperfect but useful scene appearance,
- and a renderable observation from arbitrary cameras.

The ideal system may therefore combine them:

```text
              SYMBOLIC / ENGINE STATE
                 exact facts
                     |
                     v
              object transforms
              topology / physics
                     |
                     +
                     |
                     v
              GAUSSIAN WORLD MEMORY
               visual/spatial state
                     |
                     v
              rough rendered reference
                     |
                     v
                    H3
```

The engine is deterministic truth.

The splat is visual memory.

H3 is the cinematic imagination layer.

---

# 24. Why Not Just Use Previous Video Frames?

Previous frames are useful but insufficient.

They fail when:

- the camera visits a new viewpoint,
- a long time has passed,
- an object was last visible from the wrong side,
- the new shot needs geometry hidden in prior footage,
- the room has undergone multiple changes,
- or the previous clip does not expose the required spatial relationships.

The persistent splat can render a new approximate reference from a camera angle that was never previously filmed.

That is a major advantage.

---

# 25. Why Not Require the Video Model to Remember Everything?

Because long-term persistence is fundamentally different from generation quality.

A model can be excellent at producing a coherent 10-second clip and still fail to remember:

- which shelf an object was on 100 shots ago,
- whether a window was open,
- how far a chair had moved,
- what was behind the camera,
- or which version of a room is currently canonical.

Externalizing that memory gives us:

- inspectability,
- editability,
- determinism,
- branching,
- rollback,
- explicit state,
- and model independence.

A future H3 successor can replace H3 without requiring the entire persistence architecture to be rebuilt.

---

# 26. Model Upgrades Become Replaceable Components

One of the strongest reasons to pursue this architecture is longevity.

The stack can remain:

```text
persistent world
reference library
agentic retrieval
commit logic
history
validation
LoRAs
engine integration
```

while the generative model changes:

```text
H3
 -> H4
 -> future Seedance-class model
 -> something better
```

The orchestration and world memory remain valuable.

This is one reason the system can potentially achieve capability beyond the apparent limits of any single frozen model.

---

# 27. Prototype Architecture

A practical first prototype does not need the full system.

## Phase 1: Closed Room Loop

Goal:

> Demonstrate that an H3-generated room can be reconstructed, re-rendered from a new angle, fed back to H3, and remain recognizably persistent.

Pipeline:

```text
1. Generate 5-10 second H3 room shot
2. Sample useful frames
3. Run QuerySplat
4. Render novel camera angles
5. Feed rendered views back to H3 Ref2VA
6. Generate another shot
7. Compare object/layout persistence
```

Success criterion:

- the next H3 shot preserves major room layout and object identity better with the splat-derived reference than without it.

---

## Phase 2: Controlled Object Movement

Goal:

> Update only one changed object.

Example:

```text
chair moves
```

Pipeline:

```text
canonical room splat
    |
H3 shot moves chair
    |
QuerySplat candidate
    |
register
    |
detect chair movement
    |
preserve room
    |
update chair only
```

Success criterion:

- unchanged room regions survive multiple update cycles.

---

## Phase 3: Object Removal and Addition

Test:

```text
glass on table
 -> character removes glass
 -> later returns glass
```

Validate:

- safe deletion,
- preserved table geometry,
- correct historical state,
- restoration from prior object memory.

---

## Phase 4: New Space Expansion

Test:

```text
known hallway
 -> H3 generates unseen bedroom
 -> QuerySplat reconstructs bedroom
 -> register bedroom to hallway
 -> commit
 -> revisit bedroom later
```

Success criterion:

- newly imagined space becomes persistent.

---

## Phase 5: Hybrid Engine + Splat Conditioning

Use an engine to generate:

- deterministic actor blocking,
- exact camera path,
- object transforms,
- rough room geometry.

Then combine:

```text
engine playblast
+
persistent splat render
+
identity refs
+
style refs
+
previous-shot tail
+
H3 Ref2VA
```

Measure whether this improves:

- camera control,
- object permanence,
- hand-object interactions,
- action continuity.

---

## Phase 6: Agentic Commit Decisions

Replace hand-written update selection with an agent plus deterministic tools.

Agent outputs:

```text
KEEP
RIGID_REALIGN
SIM3_REALIGN
RECOLOR
LOCAL_OPTIMIZE
PRUNE_AND_RESPLAT
FULL_REGION_RECONSTRUCT
```

The agent does not perform numerical registration itself.

It selects and orchestrates the appropriate specialized tools.

---

# 28. Evaluation Metrics

We should evaluate more than image beauty.

## Persistence

- object position drift,
- orientation drift,
- object count consistency,
- room topology consistency,
- layout stability,
- identity stability.

## Generative Quality

- visual quality,
- motion quality,
- reference fidelity,
- cinematography,
- temporal coherence.

## World Update Safety

- false deletions,
- false additions,
- incorrect geometry overwrite,
- corruption of unobserved regions,
- confidence calibration.

## Efficiency

- reconstruction latency,
- H3 generation cost,
- map update cost,
- storage growth,
- amount of scene needing reoptimization,
- reference retrieval overhead.

## Long-Horizon Robustness

Run the world through dozens or hundreds of updates and measure whether static regions drift over time.

This may be one of the most important tests.

---

# 29. Potential Data Model

A region/object representation might contain:

```yaml
region_id:
parent_region:
object_ids: []

transform:
  translation:
  rotation:
  scale:

geometry:
  gaussian_chunk:
  collision_proxy:
  bounds:

appearance:
  canonical_refs: []
  embeddings: []
  material_hints:

confidence:
  geometry:
  appearance:
  registration:
  semantic_identity:

history:
  created_at_shot:
  last_observed_shot:
  observation_count:
  versions: []

state:
  canonical:
  provisional:
  changed:
  uncertain:
```

Objects may additionally contain:

```yaml
object_id:
semantic_class:
identity_ref:
lora:
current_region:
current_transform:
state_variables:
```

Example:

```yaml
object_id: glass_04
semantic_class: crystal_tumbler
current_region: kitchen_table
state_variables:
  fill_fraction: 0.35
  contents: water
```

The splat does not need to represent every semantic property.

The symbolic layer and visual layer can coexist.

---

# 30. Separation of Truth

A useful principle is to separate multiple kinds of "truth."

## Symbolic Truth

```text
door_02 is open
glass_04 belongs to Sarah
chair_02 moved after shot 101
```

## Spatial Truth

```text
door transform
glass position
chair position
room topology
```

## Visual Truth

```text
what Sarah looks like
what the kitchen surface looks like
how the chair is textured
```

## Cinematic Truth

```text
camera movement
lighting intent
lens behavior
editing rhythm
performance
```

No single representation needs to own all four.

Possible ownership:

```text
symbolic truth  -> world database / agent state
spatial truth   -> engine + splat
visual truth    -> references + splat + LoRA
cinematic truth -> H3 + shot plan + motion references
```

---

# 31. The Larger Thesis

This system suggests a broader direction for generative media.

Instead of asking:

> "How do we make one model generate a perfectly coherent feature film?"

ask:

> "How do we give a powerful multimodal generator the same kinds of external memory, tools, retrieval, state, correction loops, and specialized subsystems that a production pipeline has?"

Then the generator no longer needs to solve:

- memory,
- world modeling,
- identity,
- exact geometry,
- editing,
- reconstruction,
- state management,
- physics,
- and continuity

all inside one forward pass.

Those capabilities can be composed.

This makes a frozen model significantly more capable at the system level.

---

# 32. Current Preferred Architecture

```text
                          STORY / USER INTENT
                                  |
                                  v
                           DIRECTOR AGENT
                                  |
                                  v
                       PERSISTENT WORLD STATE
                   +--------------+--------------+
                   |                             |
                   v                             v
             SYMBOLIC STATE               SPATIAL/VISUAL MAP
             objects/events              persistent Gaussians
             relationships               transforms/confidence
                   |                             |
                   +--------------+--------------+
                                  |
                                  v
                         REFERENCE RETRIEVAL
                   +--------------+--------------+
                   |              |              |
                   v              v              v
                Images          Videos          Audio
              identities       motion/camera    voices
              environments     prior shots      ambience
              appearance       playblasts
                   \              |              /
                    \             |             /
                     +------------+------------+
                                  |
                         optional LoRAs /
                         production adapters
                                  |
                                  v
                              H3 Ref2VA
                                  |
                                  v
                         GENERATED SHOT
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              Quality Review              3D Observation
                                            QuerySplat
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                          CHANGE ANALYSIS
                     visibility / confidence /
                     semantic / geometric diff
                                  |
                                  v
                           COMMIT DECISION
                    +-------------+-------------+
                    |                           |
                    v                           v
                  reject                     accept
              edit/regenerate           local map update
                                                |
                                                v
                                      VERSIONED WORLD STATE
```

---

# 33. The Most Important Research Question

The hardest part is probably not H3 generation or Gaussian rendering.

It is:

> **How do we decide whether a generated observation has enough evidence to modify canonical world state?**

If that decision is too permissive:

- the world slowly corrupts,
- objects disappear,
- geometry drifts,
- and hallucinations become permanent.

If it is too conservative:

- the world never evolves,
- intended changes fail to persist.

The commit algorithm therefore deserves first-class research effort.

A strong first implementation can combine:

- camera/view visibility,
- reconstruction confidence,
- multi-view agreement,
- semantic identity,
- canonical-map confidence,
- object-level state,
- change classification,
- and explicit agent reasoning.

---

# 34. Near-Term Experiments Worth Running

### Experiment A — Splat Feedback

Generate a room with H3.

QuerySplat it.

Render an unseen view.

Give that view to H3.

Measure whether H3 preserves the world better than a no-splat baseline.

---

### Experiment B — Reference Clip From Splat

Create a rough 3D camera move through the splat.

Use the rough render as H3's spatial/motion reference.

Test whether H3 upgrades it into a photorealistic or stylistically consistent shot while preserving layout.

---

### Experiment C — Engine vs Splat vs Both

Compare:

```text
H3 + image references only
H3 + engine playblast
H3 + splat render
H3 + engine playblast + splat render
```

Measure:

- geometry,
- appearance,
- motion,
- object permanence,
- identity.

---

### Experiment D — Incremental Room Mutation

Cycle:

```text
generate
 -> reconstruct
 -> commit one change
 -> render new reference
 -> generate
```

Repeat many times.

Measure cumulative drift.

---

### Experiment E — Confidence-Gated Deletion

Place an object in canonical state.

Generate shots where:

- its location is occluded,
- partially visible,
- clearly visible and empty.

Ensure the system deletes the object only in the third case.

---

### Experiment F — Reference LoRA Fallback

Find a character/object H3 does not preserve reliably with reference retrieval alone.

Train a small LoRA.

Compare:

```text
reference only
vs
reference + LoRA
```

The goal is to identify when parametric memory is actually necessary.

---

# 35. Design Principles

1. **Persistence lives outside the video model.**
2. **H3 should receive shot-specific multimodal context, not the entire world.**
3. **Unobserved space must survive.**
4. **A generated frame is evidence, not automatically truth.**
5. **Only changed regions should be rewritten.**
6. **Prefer transforms over reconstruction for rigid movement.**
7. **Separate geometry changes from appearance changes.**
8. **Keep history and deltas.**
9. **Use LoRAs for persistent learned identity/prior problems, not transient state.**
10. **Use engines for deterministic facts and blocking when useful.**
11. **Use splats as visual/spatial memory, not necessarily final rendering.**
12. **Let H3 beautify crude but geometrically useful references.**
13. **Treat reference selection as multimodal RAG.**
14. **Treat world updates as transactions.**
15. **Confidence must govern commits.**
16. **The architecture should survive model upgrades.**

---

# 36. Working Name Ideas

Not required for the architecture, but useful shorthand:

- Persistent World Splat
- Generative World Memory
- WorldSplat Memory
- Visual State Map
- Generative Minimap
- Persistent Scene Memory
- World State Splat
- Episodic Scene Graph
- Gaussian World Cache
- Persistent Imaginary-World SLAM

The phrase **"generative minimap"** is especially useful conceptually because it reinforces that the representation is intentionally sparse and functional rather than a perfect final render.

---

# 37. Current Conclusion

The current preferred direction is:

> **H3 Ref2VA + agentic multimodal retrieval + persistent Gaussian world memory + QuerySplat-style observation lifting + confidence-gated local updates + optional deterministic game-engine state + LoRA fallback for stubborn visual identity.**

The Gaussian map functions as a long-lived, editable visual/spatial memory.

H3 renders observations of that world.

QuerySplat or similar systems lift new observations back into candidate 3D.

An agentic commit layer decides what becomes canonical.

The result is a world that can persist beyond the context window and beyond any individual generated clip.

This is potentially enough to turn a very strong video model into something much closer to a **persistent generative world simulator** without requiring the base model itself to solve every aspect of long-horizon consistency internally.

The key engineering challenge is not simply generating beautiful shots.

It is making the world remember the right things.
