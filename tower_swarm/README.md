# Tower Swarm (SDL3 + WASM)

Source: `games/tower_swarm/src/`

WASM bundle:
- Host page: `frontend/webai/projects/noobygame/public/wasm/tower-swarm/index.html`
- Build script: `games/tower_swarm/build_web.ps1`

## Configuration

Every gameplay/UI tuning value from `src/Constants.h` is loaded from
`config/tower_swarm.json` (source copy: `tower_swarm/config/tower_swarm.json`).
The CMake build copies `config/` next to `TowerSwarm.exe`; the web build
preloads it into the virtual FS (`tower_swarm.data`, deploy it next to the
`.js`/`.wasm`).

- Keys are `"<section>.<name>"`, matching the C++ namespaces: `level.kBaseHp`,
  `characters.brix.kSignatureCooldownSec`, ... Values declared directly in
  `namespace tower_swarm` are under `"general"`. Colors are `[r, g, b, a]`
  (0..255). Missing keys keep their compiled-in default; unknown keys and
  wrong types are logged and ignored.
- Edits apply **live**: the game checks the file twice a second and logs
  `[tunables] reloaded ...`. A file with a syntax error is ignored (logged)
  and the previous values stay.
- **Restart required** for values only read at startup: `general.kWindowWidthPx`,
  `kWindowHeightPx`, `kWorldWidthPx`, `kWorldHeightPx`, `kTileSizePx`,
  `k*PoolCapacity`, `characters.base_stats.*` (copied into the character
  table on first use), and the whole `"engine"` section (grid, timestep,
  vsync, audio). Per-entity stats apply to entities spawned after the edit.
- A few values stay compile-time `constexpr` and are not in the JSON (array
  sizes and constexpr tables): `evolution::kVisualBandCount`,
  `characters::flara::kStage4SimultaneousTargets`, `relics::kSlotCount`,
  `inter_level_shop::kBazaarOfferCount`, the `wave_shop::k*DurationWaves`
  and all `relic_unlocks::*` costs.
- If the file is missing or cannot be parsed at startup, the game logs it and
  runs with the compiled-in defaults.

Command line (desktop):

- `--config <path>`: use another config file (default `config/tower_swarm.json`,
  resolved against the working directory, then the executable directory).
- `--write-default-config <path>`: write every tunable with its compiled-in
  default plus the `"engine"` section to `<path>` and exit (no window).
  Use it to regenerate `config/tower_swarm.json` after adding tunables.
