mod diag;
mod models;

use clap::{Parser, Subcommand};
use sentinel_core::export::ScriptExporter;
use sentinel_core::scaffold::ProjectScaffolder;
use sentinel_core::components::{Component, ComponentRegistry};
use std::collections::HashMap;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    name = "sentinel",
    author,
    version,
    about = "Sentinel ML Training Supervisor & Framework",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start daemon and launch the low-overhead Web UI
    Dev {
        #[arg(short, long, default_value = "2512")]
        port: u16,
        /// Address to listen on. Loopback by default: the API can start processes and has no authentication.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Train a graph component in the foreground (component id or path to a component YAML)
    Run {
        /// Component id (see `sentinel component list`) or a .yaml file
        component: String,
        /// Hyperparameter override, repeatable: --set lr=0.001
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Launch hyperparameter sweep
    Sweep {
        /// Path to sweep config
        config: String,
    },
    /// Component commands (list, new, check, save-run)
    Component {
        #[command(subcommand)]
        command: ComponentCommands,
    },
    /// Manage user-global templates (~/.sentinel/templates)
    Template {
        #[command(subcommand)]
        command: TemplateCommands,
    },
    /// Export run as standalone pure bare PyTorch script
    Export {
        #[arg(default_value = "script")]
        format: String,
        #[arg(short, long, default_value = "vision/image-classification@v1")]
        component: String,
        #[arg(short, long, default_value = "train_standalone.py")]
        output: String,
    },
    /// Explain model decision using axiomatic Integrated Gradients & SVD Spectral Rank
    Explain {
        /// Run id (defaults to most recent run with an exported model)
        #[arg(short, long)]
        run: Option<String>,
        /// Feature vector: --input "5.1,3.5,1.4,0.2"
        #[arg(short, long = "input", value_name = "V1,V2,...")]
        input: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Add/install a modular neural block or template (e.g. `sentinel add --module vit-tiny`)
    Add {
        /// Module name to install (e.g. vit-tiny, unet, residual-mlp)
        #[arg(long = "module")]
        module: String,
        /// Install globally into ~/.sentinel/templates/
        #[arg(long)]
        global: bool,
    },
    /// Use trained models: list exports, predict on new data, copy weights/TorchScript/ONNX out
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
    /// One-shot system vitals: CPU, memory, swap, load, GPUs, and the daemon's jobs
    Status {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
        #[arg(short, long, default_value = "2512")]
        port: u16,
    },
    /// Live vitals, refreshed in place (uses the daemon's per-job view when it is running)
    Top {
        /// Seconds between refreshes
        #[arg(short, long, default_value = "1")]
        interval: f32,
        #[arg(short, long, default_value = "2512")]
        port: u16,
        /// How many of the busiest processes to list (0 hides the list)
        #[arg(long, default_value = "8")]
        procs: usize,
    },
    /// List GPUs with live utilization, memory, temperature and power (nvidia-smi on NVIDIA machines)
    Gpus {
        #[arg(long)]
        json: bool,
    },
    /// Check Python, PyTorch, GPU drivers, ports and project setup, with fixes for this OS
    Doctor {
        #[arg(short, long, default_value = "2512")]
        port: u16,
    },
    /// Initialize a new Sentinel project
    New {
        /// Project name
        name: String,
    },
    /// Convert/initialize an existing directory into a Sentinel repository
    Init {
        /// Directory path (defaults to current directory ".")
        #[arg(default_value = ".")]
        path: String,
    },
}

#[derive(Subcommand, Debug)]
enum ComponentCommands {
    /// List all built-in and custom components
    List,
    /// Create a new custom component starter
    New {
        name: String,
        #[arg(short, long, default_value = "custom")]
        domain: String,
        #[arg(short, long)]
        extends: Option<String>,
    },
    /// Validate a component YAML
    Check { path: String },
    /// Save an existing run as a reusable component
    SaveRun { run_id: String, name: String },
    /// List the neural-network building blocks available to component graphs
    Blocks,
    /// List the tasks, losses, optimizers, schedulers and training strategies a component's `training:` section accepts
    Training {
        /// Show only this task (classification, binary, multilabel, regression, ranking, reconstruction, vae, custom)
        #[arg(long)]
        task: Option<String>,
    },
    /// Show a graph component: execution order and the tensor shape on every port
    Show {
        component: String,
        /// Parameter override, repeatable: --set hidden=64
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Compile a graph component into a standalone PyTorch script
    Compose {
        component: String,
        #[arg(short, long, default_value = "train.py")]
        output: String,
        /// Parameter override, repeatable: --set hidden=64
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Emit only the `Model` class (no data loading or training loop)
        #[arg(long)]
        model_only: bool,
    },
}

#[derive(Subcommand, Debug)]
enum TemplateCommands {
    /// List all user-global templates (~/.sentinel/templates)
    List,
    /// Add/install a component YAML file as a user-global template
    Add {
        /// Path to the component YAML file to install
        path: String,
        /// Force overwrite if template already exists
        #[arg(short, long)]
        force: bool,
    },
    /// Remove a user-global template by component ID
    Remove {
        /// Component ID (e.g. tabular/my-template@v1)
        id: String,
    },
    /// Print the user templates directory path
    Dir,
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    /// Runs that exported a model
    List,
    /// Predict from a run's TorchScript model: --input "5.1,3.5" (repeat for several rows)
    Predict {
        /// Run id or any unambiguous prefix (see `sentinel model list`)
        run: String,
        #[arg(short, long = "input", value_name = "V1,V2,...", required = true)]
        input: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Draw samples from a GAN run's generator
    Generate {
        run: String,
        #[arg(short, long, default_value = "5")]
        count: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        #[arg(long)]
        json: bool,
    },
    /// Copy an exported file out of a run
    Export {
        run: String,
        /// torchscript | onnx | weights | meta | script
        #[arg(short, long, default_value = "torchscript")]
        format: String,
        /// Destination (default: ./<run>-<file>)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Explain model decision using axiomatic Integrated Gradients & SVD Spectral Rank
    Explain {
        /// Run id or prefix
        run: String,
        /// Feature vector: --input "5.1,3.5,1.4,0.2"
        #[arg(short, long = "input", value_name = "V1,V2,...")]
        input: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

/// Rust ignores SIGPIPE, so `println!` panics when stdout is a closed pipe (`sentinel model list | head`). A CLI should
/// exit quietly like every other Unix tool.
#[cfg(unix)]
fn restore_sigpipe() {
    // SAFETY: resets one signal's disposition to the OS default before any threads or output exist.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    #[cfg(unix)]
    restore_sigpipe();
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Dev { port, host }) => {
            run_dev(port, &host).await?;
        }
        Some(Commands::Run { component, set }) => {
            let built = build_any(&component, &set)?;
            let code = built.training_script(&component)?;
            let root = std::env::current_dir()?;
            let script = sentinel_core::sdk::write_generated_script(&root, &component, &uuid::Uuid::new_v4().simple().to_string()[..8], &code)?;
            println!("Running {component}  (script: {})", script.display());
            let python = sentinel_core::platform::python_exe();
            let status = tokio::process::Command::new(&python)
                .arg(&script)
                .env("PYTHONPATH", sentinel_core::sdk::python_path(&root)?)
                .status()
                .await
                .map_err(|e| anyhow::anyhow!("could not start {python}: {e}"))?;
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Some(Commands::Model { command }) => models::run(command)?,
        Some(Commands::Explain { run, input, json }) => {
            let run_id = match run {
                Some(r) => r,
                None => {
                    let root = std::env::current_dir().unwrap_or_default().join("mlruns");
                    sentinel_core::tracking::list_runs(&root)
                        .into_iter()
                        .find(|r| {
                            sentinel_core::tracking::run_dir(&root, &r.run_id)
                                .map(|d| sentinel_core::predict::read_meta(&d).is_some())
                                .unwrap_or(false)
                        })
                        .map(|r| r.run_id)
                        .ok_or_else(|| anyhow::anyhow!("no trained run with exported model found (use `sentinel run <component>` first)"))?
                }
            };
            models::run(ModelCommands::Explain { run: run_id, input, json })?;
        }
        Some(Commands::Add { module, global }) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let registry = ComponentRegistry::full(&cwd);
            
            // 1. Query Sentinel Hub CDN registry
            let cdn_base = "https://raw.githubusercontent.com/rimraf-adi/sentinel-hub/main";
            let mut resolved_yaml: Option<(String, String)> = None;

            let reg_output = std::process::Command::new("curl")
                .args(["-sSL", "-H", "Cache-Control: no-cache", "--max-time", "3", &format!("{cdn_base}/registry.json")])
                .output();

            if let Ok(out) = reg_output {
                if out.status.success() {
                    if let Ok(manifest) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                        if let Some(modules) = manifest["modules"].as_array() {
                            let matched = modules.iter().find(|m| {
                                m["id"].as_str() == Some(&module)
                                    || m["component_id"].as_str() == Some(&module)
                                    || m["name"].as_str().map(|s| s.to_lowercase()) == Some(module.to_lowercase())
                            });
                            if let Some(m) = matched {
                                if let Some(path) = m["path"].as_str() {
                                    let mod_url = format!("{cdn_base}/{path}");
                                    let mod_out = std::process::Command::new("curl")
                                        .args(["-sSL", "-H", "Cache-Control: no-cache", "--max-time", "5", &mod_url])
                                        .output();
                                    if let Ok(mo) = mod_out {
                                        if mo.status.success() && !mo.stdout.is_empty() {
                                            if let Ok(content) = String::from_utf8(mo.stdout) {
                                                resolved_yaml = Some((content, "sentinel-hub (https://github.com/rimraf-adi/sentinel-hub)".to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 2. Offline / local fallback if CDN unavailable or module is built-in
            let (yaml_content, origin) = match resolved_yaml {
                Some(pair) => pair,
                None => {
                    let matched_id = registry.list().into_iter().find(|c| {
                        c.component_id == module || c.component_id.contains(&module) || c.name.to_lowercase() == module.to_lowercase()
                    }).map(|c| c.component_id.clone());

                    if let Some(ref id) = matched_id {
                        let comp = registry.get(id).ok_or_else(|| anyhow::anyhow!("component not found in local registry"))?;
                        (serde_yaml::to_string(&comp)?, "built-in offline cache".to_string())
                    } else {
                        anyhow::bail!("module '{module}' not found in sentinel-hub or local built-ins. Run `sentinel component list` to see available local components.");
                    }
                }
            };

            let target_dir = if global {
                ComponentRegistry::user_templates_dir()
            } else {
                cwd.join("components")
            };
            std::fs::create_dir_all(&target_dir)?;
            let file_name = if module.ends_with(".yaml") { module.clone() } else { format!("{module}.yaml") };
            let target_file = target_dir.join(&file_name);
            std::fs::write(&target_file, &yaml_content)?;

            println!("✨ Successfully added module to {}", target_file.display());
            println!("   Origin:      {origin}");
            println!("   Inspect DAG: http://localhost:2512");
            println!("   Train:       sentinel run {}", target_file.display());
        }
        Some(Commands::Status { json, port }) => diag::status(json, port)?,
        Some(Commands::Top { interval, port, procs }) => diag::top(interval, port, procs).await?,
        Some(Commands::Gpus { json }) => diag::gpus(json)?,
        Some(Commands::Doctor { port }) => diag::doctor(port)?,
        Some(Commands::Sweep { config }) => {
            println!("🚀 Launching Sentinel Hyperparameter Sweep with config: {config}");
            let mut cmd = tokio::process::Command::new("uv");
            cmd.arg("run")
                .arg("python")
                .arg("recipes/sweep_orchestrator.py")
                .arg("--trials")
                .arg("8")
                .arg("--workers")
                .arg("2");

            let status = cmd.status().await?;
            if !status.success() {
                eprintln!("Sweep execution exited with error code: {status}");
                std::process::exit(1);
            }
        }
        Some(Commands::Component { command }) => match command {
            ComponentCommands::List => {
                let cwd = std::env::current_dir().unwrap_or_default();
                let registry = ComponentRegistry::full(&cwd);
                let list = registry.list();
                let mut builtin_cnt = 0;
                let mut user_cnt = 0;
                let mut project_cnt = 0;
                for t in &list {
                    match t.origin.as_deref() {
                        Some("user") => user_cnt += 1,
                        Some("project") => project_cnt += 1,
                        _ => builtin_cnt += 1,
                    }
                }

                println!(
                    "Found {} components ({} built-in, {} user template(s), {} project):",
                    list.len(),
                    builtin_cnt,
                    user_cnt,
                    project_cnt,
                );
                for t in list {
                    let parent = t
                        .extends
                        .as_deref()
                        .map(|p| format!(" (extends {p})"))
                        .unwrap_or_default();
                    let origin_badge = match t.origin.as_deref() {
                        Some("user") => "[user template]",
                        Some("project") => "[project]",
                        _ => "[built-in]",
                    };
                    println!(
                        "  - {:<36} {:<16} [{}] {}{}",
                        t.component_id, origin_badge, t.domain, t.name, parent
                    );
                }
            }
            ComponentCommands::New {
                name,
                domain,
                extends,
            } => {
                let yaml = ComponentRegistry::create_custom_starter(
                    &name,
                    &domain,
                    extends.as_deref(),
                )?;
                let file_name = format!("{}.yaml", name.to_lowercase().replace(' ', "-"));
                let out_dir = Path::new("components");
                std::fs::create_dir_all(out_dir)?;
                let out_path = out_dir.join(&file_name);
                std::fs::write(&out_path, &yaml)?;
                println!(
                    "Created custom component starter at: {}",
                    out_path.display()
                );
            }
            ComponentCommands::Check { path } => {
                let content = std::fs::read_to_string(&path)?;
                match Component::from_yaml(&content) {
                    Ok(component) => {
                        println!("Component '{}' is VALID:", component.component_id);
                        println!("  Name: {}", component.name);
                        println!("  Domain: {}", component.domain);
                        println!("  Parameters defined: {}", component.parameters.len());
                        println!("  Locked fields: {}", component.locked.len());
                    }
                    Err(e) => {
                        eprintln!("Component validation FAILED: {e}");
                        std::process::exit(1);
                    }
                }
            }
            ComponentCommands::SaveRun { run_id, name } => {
                println!("Saving run {run_id} as component '{name}'...");
            }
            ComponentCommands::Training { task } => print_training_catalog(task.as_deref()),
            ComponentCommands::Blocks => {
                for e in sentinel_core::compound::BlockLibrary::reload_project(Path::new("blocks")) {
                    eprintln!("warning: ignoring compound block: {e}");
                }
                let catalog = sentinel_core::compound::BlockLibrary::global().catalog();
                for tier in ["fundamental", "compound"] {
                    println!(
                        "\n{}",
                        if tier == "fundamental" { "FUNDAMENTAL BLOCKS  (native, one PyTorch module each)" } else { "COMPOUND BLOCKS  (subgraphs built from the blocks below them)" }
                    );
                    let mut family = String::new();
                    for i in catalog.iter().filter(|i| i.tier == tier) {
                        if i.family != family {
                            family = i.family.clone();
                            println!("\n  [{family}]");
                        }
                        let params: Vec<String> = i.params.iter().map(|p| if p.required { format!("{}*", p.name) } else { p.name.clone() }).collect();
                        println!("    {:<22} in: {:<8} out: {:<9} params: {}", i.name, i.inputs.join(","), i.outputs.join(","), params.join(", "));
                        if i.uses.is_empty() {
                            println!("        {}", i.description);
                        } else {
                            println!("        {}  (built from: {})", i.description, i.uses.join(", "));
                        }
                    }
                }
                println!("\n(* = required parameter)");
            }
            ComponentCommands::Show { component, set } => {
                use sentinel_core::components::Built;
                let show = |title: Option<&str>, g: &sentinel_core::graph::ValidatedGraph| {
                    if let Some(t) = title {
                        println!("\n== {t} ==");
                    }
                    for (name, shape) in &g.spec.inputs {
                        println!("input  {name:<14} {}", sentinel_core::graph::fmt_shape(shape));
                    }
                    print_nodes(g, 0);
                    for (name, src) in &g.outputs {
                        println!("output {name:<14} = {}", src.key());
                    }
                };
                match build_any(&component, &set)? {
                    Built::Graph(g) => {
                        show(None, &g);
                        print_summary("MODEL", &g.stats());
                    }
                    Built::Gan(v) => {
                        show(Some("generator"), &v.generator);
                        show(Some("discriminator"), &v.discriminator);
                        print_summary("GENERATOR", &v.generator.stats());
                        print_summary("DISCRIMINATOR", &v.discriminator.stats());
                        let total = v.generator.stats().params + v.discriminator.stats().params;
                        println!("\nGAN total     {} parameters ({} as float32)", count(total), size(total * 4));
                        println!("\ntraining: latent {} · {} epochs · lr_g {} · lr_d {} · {} discriminator step(s) per generator step", v.training.latent, v.training.epochs, v.training.lr_g, v.training.lr_d, v.training.d_steps);
                    }
                }
            }
            ComponentCommands::Compose { component, output, set, model_only } => {
                let built = build_any(&component, &set)?;
                let code = if model_only {
                    format!("import torch\nimport torch.nn as nn\n\n{}", built.model_code())
                } else {
                    built.training_script(&component)?
                };
                std::fs::write(&output, code)?;
                println!("Composed '{component}' -> {output}");
                println!("Run anywhere with PyTorch: python {output}");
            }
        },
        Some(Commands::Template { command }) => match command {
            TemplateCommands::Dir => {
                println!("{}", ComponentRegistry::user_templates_dir().display());
            }
            TemplateCommands::List => {
                let dir = ComponentRegistry::user_templates_dir();
                let mut reg = ComponentRegistry::new();
                let loaded = reg.load_from_dir_with_origin(&dir, Some("user")).unwrap_or(0);
                println!("User templates directory: {}", dir.display());
                if loaded == 0 {
                    println!("No user templates found. Install one with: sentinel template add <file.yaml>");
                } else {
                    println!("Found {} user template(s):", loaded);
                    for t in reg.list() {
                        if t.origin.as_deref() == Some("user") {
                            println!("  * {:<36} [{}] {}", t.component_id, t.domain, t.name);
                        }
                    }
                }
            }
            TemplateCommands::Add { path, force } => {
                let p = Path::new(&path);
                if !p.exists() {
                    anyhow::bail!("File not found: {path}");
                }
                let content = std::fs::read_to_string(p)?;
                let c = Component::from_yaml(&content)?;

                if ComponentRegistry::new().get(&c.component_id).is_some() {
                    anyhow::bail!(
                        "Cannot install template '{}': conflicts with built-in component. Built-in components cannot be shadowed.",
                        c.component_id
                    );
                }

                let dir = ComponentRegistry::user_templates_dir().join(&c.domain);
                std::fs::create_dir_all(&dir)?;
                let slug = sentinel_core::authoring::slugify(&c.name);
                let dest = dir.join(format!("{slug}.yaml"));
                if dest.exists() && !force {
                    anyhow::bail!("Template already exists at {}. Use --force to overwrite.", dest.display());
                }
                std::fs::write(&dest, &content)?;
                println!("✨ Installed template '{}' -> {}", c.component_id, dest.display());
            }
            TemplateCommands::Remove { id } => {
                let dir = ComponentRegistry::user_templates_dir();
                if !dir.exists() {
                    anyhow::bail!("No user templates directory at {}", dir.display());
                }
                fn walk_remove(d: &Path, target_id: &str) -> std::io::Result<bool> {
                    if !d.is_dir() { return Ok(false); }
                    for entry in std::fs::read_dir(d)? {
                        let path = entry?.path();
                        if path.is_file() {
                            if let Ok(c) = std::fs::read_to_string(&path) {
                                if let Ok(t) = Component::from_yaml(&c) {
                                    if t.component_id == target_id || path.file_stem().and_then(|s| s.to_str()) == Some(target_id) {
                                        std::fs::remove_file(&path)?;
                                        return Ok(true);
                                    }
                                }
                            }
                        } else if path.is_dir() {
                            if walk_remove(&path, target_id)? {
                                return Ok(true);
                            }
                        }
                    }
                    Ok(false)
                }
                if walk_remove(&dir, &id)? {
                    println!("✨ Removed template '{id}' from {}", dir.display());
                } else {
                    anyhow::bail!("Template '{id}' not found in {}", dir.display());
                }
            }
        },
        Some(Commands::Export { format, component, output }) => {
            println!("Exporting component '{component}' as pure standalone {format} to: {output}");
            let cwd = std::env::current_dir().unwrap_or_default();
            let registry = ComponentRegistry::full(&cwd);

            let tmpl = registry.get(&component).ok_or_else(|| {
                anyhow::anyhow!("Component '{}' not found in registry", component)
            })?;

            let resolved = tmpl.resolve_config(&HashMap::new())?;
            let python_script = ScriptExporter::export_pytorch_script(tmpl, &resolved);

            std::fs::write(&output, python_script)?;
            println!("✨ Successfully exported standalone PyTorch script to: {output}");
            println!("💡 Run anywhere without Sentinel: python {output}");
        }
        Some(Commands::New { name }) => {
            let target_path = Path::new(&name);
            if target_path.exists() {
                eprintln!("Error: Directory '{}' already exists.", name);
                std::process::exit(1);
            }
            println!("Scaffolding new Sentinel project in: {}", target_path.display());
            ProjectScaffolder::scaffold(target_path, &name)?;
            println!("✨ Successfully created project '{name}'!");
            println!("\nNext steps:");
            println!("  cd {name}");
            println!("  sentinel component list");
            println!("  sentinel dev");
        }
        Some(Commands::Init { path }) => {
            let target_path = Path::new(&path);
            let canonical = target_path.canonicalize().unwrap_or_else(|_| target_path.to_path_buf());
            let project_name = canonical
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project");

            println!("Initializing Sentinel in: {}", target_path.display());
            ProjectScaffolder::init(target_path, project_name)?;
            println!("✨ Successfully converted '{project_name}' into a Sentinel repository!");
            println!("   Generated: sentinel.lock (lockfile), sentinel.yaml, and machines.yaml");
            println!("\nNext steps:");
            println!("   sentinel component list");
            println!("   sentinel dev");
        }
        None => {
            // Default run on port 2512
            run_dev(2512, "127.0.0.1").await?;
        }
    }

    Ok(())
}

/// Resolve a registry component (plus `--set k=v` overrides) into a validated graph.
/// Resolve a component (id or YAML file) plus `--set` overrides into a graph, or a GAN's pair of graphs.
fn build_any(component: &str, set: &[String]) -> anyhow::Result<sentinel_core::components::Built> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let _ = sentinel_core::compound::BlockLibrary::reload_project(&cwd.join("blocks"));
    let registry = ComponentRegistry::full(&cwd);
    let from_file;
    let c = if component.ends_with(".yaml") || component.ends_with(".yml") {
        from_file = Component::from_yaml(&std::fs::read_to_string(component)?)?;
        &from_file
    } else {
        registry
            .get(component)
            .ok_or_else(|| anyhow::anyhow!("Component '{component}' not found (see `sentinel component list`)"))?
    };
    let mut overrides = HashMap::new();
    for kv in set {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--set expects KEY=VALUE, got '{kv}'"))?;
        overrides.insert(k.to_string(), serde_yaml::from_str(v)?);
    }
    c.validate_overrides(&overrides)?;
    Ok(c.build(&overrides)?)
}

fn count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn human(n: f64) -> String {
    if n >= 1e9 { format!("{:.2} G", n / 1e9) } else if n >= 1e6 { format!("{:.2} M", n / 1e6) } else if n >= 1e3 { format!("{:.1} k", n / 1e3) } else { format!("{n:.0}") }
}

fn size(bytes: u64) -> String {
    if bytes >= 1 << 20 { format!("{:.1} MB", bytes as f64 / (1 << 20) as f64) } else { format!("{:.1} KB", bytes as f64 / 1024.0) }
}

/// One summary block: parameters, memory, compute, composition and where the weights live.
fn print_summary(title: &str, st: &sentinel_core::graph::GraphStats) {
    println!("\n{title}");
    println!("  parameters   {} ({} as float32)", count(st.params), size(st.bytes_fp32));
    match st.macs {
        Some(m) => println!("  compute      {}MACs per sample (matrix and convolution layers)", human(m as f64)),
        None => println!("  compute      unknown (some sizes are symbolic)"),
    }
    println!("  structure    {} fundamental blocks · {} compound blocks · {} level(s) of nesting", st.blocks, st.compounds, st.depth);
    let mut by: Vec<(&String, &u64)> = st.by_block.iter().filter(|(_, v)| **v > 0).collect();
    by.sort_by(|a, b| b.1.cmp(a.1));
    if !by.is_empty() {
        let parts: Vec<String> = by.iter().take(5).map(|(k, v)| format!("{k} {:.0}%", **v as f64 / st.params.max(1) as f64 * 100.0)).collect();
        println!("  weights in   {}", parts.join(" · "));
    }
}

/// Print a graph's nodes with shapes; compound blocks are expanded one indent level deeper.
fn print_nodes(g: &sentinel_core::graph::ValidatedGraph, depth: usize) {
    let pad = "  ".repeat(depth);
    let st = g.stats();
    for node in &g.order {
        let b = g.kind(node);
        let ns = &st.by_node[node];
        let tag = if ns.shared { "shared".to_string() } else if ns.params > 0 { count(ns.params) } else { String::new() };
        for port in b.outputs() {
            let shape = &g.shapes[&format!("{node}.{port}")];
            let ins: Vec<String> = b.inputs().iter().map(|p| g.wires[&(node.clone(), p.clone())].key()).collect();
            println!(
                "node   {pad}{:<14} {:<22} {:<20} {:>11}  <- {}",
                format!("{node}.{port}"),
                if b.tier() == "compound" { format!("{} (compound)", b.name()) } else { b.name() },
                sentinel_core::graph::fmt_shape(shape),
                if port == b.outputs()[0] { tag.clone() } else { String::new() },
                ins.join(", ")
            );
        }
        if let Some(inner) = g.compounds.get(node) {
            print_nodes(inner, depth + 1);
        }
    }
}

async fn run_dev(port: u16, host: &str) -> anyhow::Result<()> {
    let supervisor = match sentinel_daemon::store::JobStore::open(".sentinel/jobs") {
        Ok(store) => sentinel_daemon::supervisor::ProcessSupervisor::with_store(store),
        Err(e) => {
            eprintln!("warning: job history will not be saved ({e})");
            sentinel_daemon::supervisor::ProcessSupervisor::new()
        }
    };
    let mut monitor = sentinel_daemon::scheduler::HardwareMonitor::new();
    // The pool holds the GPUs that really exist (NVIDIA, via nvidia-smi); CPU-only machines get an empty pool.
    let gpu_pool = sentinel_daemon::scheduler::DynamicGpuPool::new(monitor.schedulable_gpus());
    let hardware_monitor = std::sync::Arc::new(tokio::sync::Mutex::new(monitor));
    let cwd = std::env::current_dir().unwrap_or_default();
    let registry = ComponentRegistry::full(&cwd);
    let components = std::sync::Arc::new(registry);

    let state = sentinel_daemon::server::AppState {
        vitals: Default::default(),
        ws_clients: Default::default(),
        supervisor,
        gpu_pool,
        hardware_monitor,
        components,
    };

    sentinel_daemon::server::spawn_sampler(state.clone());
    let app = sentinel_daemon::server::Server::router(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("🚀 Sentinel Daemon active at http://{}:{port}", if host == "0.0.0.0" { "localhost" } else { host });
    if host != "127.0.0.1" && host != "localhost" && host != "::1" {
        println!("⚠️  Listening on {host}: anyone who can reach this port can start processes on this machine (no authentication).");
    }
    println!("   API health: http://localhost:{port}/api/health");
    println!("   WebSocket telemetry: ws://localhost:{port}/ws/live");

    let is_sentinel = Path::new("sentinel.lock").exists();
    if !is_sentinel {
        println!("⚠️  Workspace: 'sentinel.lock' not found (not a Sentinel repository).");
        println!("   Run 'sentinel init' to initialize and convert this repository.");
    }
    axum::serve(listener, app).await?;
    Ok(())
}

fn print_params(params: &[serde_json::Value]) -> String {
    params.iter().map(|p| format!("{}={}", p["name"].as_str().unwrap_or("?"), p["default"])).collect::<Vec<_>>().join(", ")
}

/// Print the training catalog (the same data the UI form is built from).
fn print_training_catalog(only: Option<&str>) {
    let c = sentinel_core::training::catalog();
    let tasks = c["tasks"].as_array().cloned().unwrap_or_default();
    if let Some(t) = only {
        if !tasks.iter().any(|x| x["name"] == t) {
            let names: Vec<&str> = tasks.iter().filter_map(|x| x["name"].as_str()).collect();
            eprintln!("unknown task '{t}' (known: {})", names.join(", "));
            std::process::exit(1);
        }
    }
    println!("TASKS AND LOSSES  (the task is inferred from `loss`; set `task:` to override)");
    for t in tasks.iter().filter(|t| only.map_or(true, |o| t["name"] == o)) {
        let metrics: Vec<&str> = t["metrics"].as_array().map(|a| a.iter().filter_map(|m| m.as_str()).collect()).unwrap_or_default();
        println!("\n  {}  -  {}", t["name"].as_str().unwrap_or(""), t["help"].as_str().unwrap_or(""));
        println!("      metrics: {}", metrics.join(", "));
        for l in t["losses"].as_array().into_iter().flatten() {
            println!("      loss {:<14} {}", l["name"].as_str().unwrap_or(""), l["help"].as_str().unwrap_or(""));
            let ps = print_params(l["params"].as_array().map(|v| v.as_slice()).unwrap_or(&[]));
            if !ps.is_empty() {
                println!("           settings: {ps}");
            }
        }
    }
    if only.is_some() {
        return;
    }
    for (title, key) in [("OPTIMIZERS", "optimizers"), ("LR SCHEDULERS  (add warmup_epochs to any)", "schedulers")] {
        println!("\n{title}");
        for o in c[key].as_array().into_iter().flatten() {
            let ps = print_params(o["params"].as_array().map(|v| v.as_slice()).unwrap_or(&[]));
            println!("  {:<16} {}{}", o["name"].as_str().unwrap_or(""), o["help"].as_str().unwrap_or(""), if ps.is_empty() { String::new() } else { format!("  [{ps}]") });
        }
    }
    println!("\nSTRATEGIES");
    for s in c["strategies"].as_array().into_iter().flatten() {
        println!("  {:<15} {}", s["name"].as_str().unwrap_or(""), s["help"].as_str().unwrap_or(""));
    }
}
