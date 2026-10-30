# Sentinel dashboard (React)

Vite + React + TypeScript, with [React Flow](https://reactflow.dev) for the component-graph builder.
The production build is embedded into the `sentinel` binary (see `crates/sentinel-daemon/src/assets.rs`),
so end users need no Node.

## Develop

```bash
cargo run -p sentinel-cli -- dev      # daemon + API on :2512 (run inside any Sentinel project)
cd ui && npm install && npm run dev   # hot-reload UI on :5173, API proxied to :2512
```

Debug builds of the daemon read `ui/dist` from disk, so `npm run build` is enough to refresh the
embedded UI without recompiling Rust.

## Build

`cargo build` runs `crates/sentinel-daemon/build.rs`, which builds `ui/dist` with npm when it is
missing or older than `ui/src`. Set `SENTINEL_SKIP_UI=1` to skip that (CI does, after building the UI once).
Without Node the build still succeeds and the dashboard shows a "not built" page; the REST API is unaffected.

```bash
cd ui && npm ci && npm run build      # typechecks, then writes ui/dist
```
