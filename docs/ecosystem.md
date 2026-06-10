# Optrix Ecosystem

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PRISMKIT                               │
│                 (Examples & Tutorials)                       │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ voxclad  │ spectune │ novastm  │ celatrix │    optrix       │
│ (API)    │ (Bench)  │ (Stream) │ (Sched)  │   (Core)        │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│               AMD ROCm / HIP Runtime                        │
├─────────────────────────────────────────────────────────────┤
│            AMD GPU Hardware (RDNA / CDNA)                   │
└─────────────────────────────────────────────────────────────┘
```

## Dependency Graph

- **optrix** — standalone core (no ecosystem deps)
- **celatrix** — depends on optrix
- **novastm** — depends on optrix
- **spectune** — depends on optrix
- **voxclad** — depends on optrix + celatrix
- **prismkit** — depends on all of the above

## Getting Started

```bash
pip install optrix celatrix novastm spectune voxclad
git clone https://github.com/kevin-wangs/prismkit.git
cd prismkit/examples
python vector_add.py
```
