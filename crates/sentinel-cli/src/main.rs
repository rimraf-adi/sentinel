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
    },
    /// Run a training job from a component or config
    Run {
        /// Path to config or component
        config: String,
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
    /// Export run as standalone pure bare PyTorch script
    Export {
        #[arg(default_value = "script")]
        format: String,
        #[arg(short, long, default_value = "vision/image-classification@v1")]
        component: String,
        #[arg(short, long, default_value = "train_standalone.py")]
        output: String,
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Dev { port }) => {
            run_dev(port).await?;
        }
        Some(Commands::Run { config }) => {
            println!("Launching Sentinel run with config: {config}");
        }
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
                let mut registry = ComponentRegistry::new();
                let components_dir = Path::new("components");
                let custom_loaded = registry.load_from_dir(components_dir).unwrap_or(0);

                println!(
                    "Found {} components ({} custom loaded):",
                    registry.list().len(),
                    custom_loaded
                );
                for t in registry.list() {
                    let parent = t
                        .extends
                        .as_deref()
                        .map(|p| format!(" (extends {p})"))
                        .unwrap_or_default();
                    println!(
                        "  - {:<32} [{}] {}{}",
                        t.component_id, t.domain, t.name, parent
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
            ComponentCommands::Blocks => {
                let mut family = "";
                for b in sentinel_core::blocks::Block::all() {
                    let i = b.info();
                    if i.family != family {
                        family = i.family;
                        println!("\n[{family}]");
                    }
                    let params: Vec<String> = i
                        .params
                        .iter()
                        .map(|p| if p.required { format!("{}*", p.name) } else { p.name.to_string() })
                        .collect();
                    println!(
                        "  {:<20} in: {:<8} out: {:<9} params: {}",
                        i.name,
                        i.inputs.join(","),
                        i.outputs.join(","),
                        params.join(", ")
                    );
                    println!("      {}", i.description);
                }
                println!("\n(* = required parameter)");
            }
            ComponentCommands::Show { component, set } => {
                let g = build_graph(&component, &set)?;
                for (name, shape) in &g.spec.inputs {
                    println!("input  {name:<14} {}", sentinel_core::graph::fmt_shape(shape));
                }
                for node in &g.order {
                    let b = g.block(node);
                    for port in b.outputs() {
                        let shape = &g.shapes[&format!("{node}.{port}")];
                        let ins: Vec<String> = b
                            .inputs()
                            .iter()
                            .map(|p| g.wires[&(node.clone(), p.to_string())].key())
                            .collect();
                        println!(
                            "node   {:<14} {:<20} {:<22} <- {}",
                            format!("{node}.{port}"),
                            b.name(),
                            sentinel_core::graph::fmt_shape(shape),
                            ins.join(", ")
                        );
                    }
                }
                for (name, src) in &g.outputs {
                    println!("output {name:<14} = {}", src.key());
                }
            }
            ComponentCommands::Compose { component, output, set, model_only } => {
                let g = build_graph(&component, &set)?;
                let code = if model_only {
                    format!("import torch\nimport torch.nn as nn\n\n{}", sentinel_core::codegen::generate_model(&g))
                } else {
                    sentinel_core::codegen::generate_training_script(&g, &component)?
                };
                std::fs::write(&output, code)?;
                println!("Composed '{component}' -> {output}");
                println!("Run anywhere with PyTorch: python {output}");
            }
        },
        Some(Commands::Export { format, component, output }) => {
            println!("Exporting component '{component}' as pure standalone {format} to: {output}");
            let mut registry = ComponentRegistry::new();
            let _ = registry.load_from_dir(Path::new("components"));

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
            run_dev(2512).await?;
        }
    }

    Ok(())
}

/// Resolve a registry component (plus `--set k=v` overrides) into a validated graph.
fn build_graph(component: &str, set: &[String]) -> anyhow::Result<sentinel_core::graph::ValidatedGraph> {
    let mut registry = ComponentRegistry::new();
    let _ = registry.load_from_dir(Path::new("components"));
    let c = registry
        .get(component)
        .ok_or_else(|| anyhow::anyhow!("Component '{component}' not found (see `sentinel component list`)"))?;
    let mut overrides = HashMap::new();
    for kv in set {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--set expects KEY=VALUE, got '{kv}'"))?;
        overrides.insert(k.to_string(), serde_yaml::from_str(v)?);
    }
    c.validate_overrides(&overrides)?;
    Ok(c.build_graph(&overrides)?)
}

async fn run_dev(port: u16) -> anyhow::Result<()> {
    let supervisor = sentinel_daemon::supervisor::ProcessSupervisor::new();
    let gpu_pool = sentinel_daemon::scheduler::DynamicGpuPool::new(4);
    let hardware_monitor = std::sync::Arc::new(tokio::sync::Mutex::new(
        sentinel_daemon::scheduler::HardwareMonitor::new(),
    ));
    let mut registry = ComponentRegistry::new();
    let _ = registry.load_from_dir(Path::new("components"));
    let components = std::sync::Arc::new(registry);

    let state = sentinel_daemon::server::AppState {
        supervisor,
        gpu_pool,
        hardware_monitor,
        components,
    };

    let app = sentinel_daemon::server::Server::router(state);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("🚀 Sentinel Daemon active at http://localhost:{port}");
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
