use clap::{Parser, Subcommand};
use sentinel_core::export::ScriptExporter;
use sentinel_core::scaffold::ProjectScaffolder;
use sentinel_core::templates::{Template, TemplateRegistry};
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
    /// Run a training job from a template or config
    Run {
        /// Path to config or template
        config: String,
    },
    /// Launch hyperparameter sweep
    Sweep {
        /// Path to sweep config
        config: String,
    },
    /// Template commands (list, new, check, save-run)
    Template {
        #[command(subcommand)]
        command: TemplateCommands,
    },
    /// Export run as standalone pure bare PyTorch script
    Export {
        #[arg(default_value = "script")]
        format: String,
        #[arg(short, long, default_value = "vision/image-classification@v1")]
        template: String,
        #[arg(short, long, default_value = "train_standalone.py")]
        output: String,
    },
    /// Initialize a new Sentinel project
    New {
        /// Project name
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum TemplateCommands {
    /// List all built-in and custom templates
    List,
    /// Create a new custom template starter
    New {
        name: String,
        #[arg(short, long, default_value = "custom")]
        domain: String,
        #[arg(short, long)]
        extends: Option<String>,
    },
    /// Validate a template YAML
    Check { path: String },
    /// Save an existing run as a reusable template
    SaveRun { run_id: String, name: String },
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
        Some(Commands::Template { command }) => match command {
            TemplateCommands::List => {
                let mut registry = TemplateRegistry::new();
                let templates_dir = Path::new("templates");
                let custom_loaded = registry.load_from_dir(templates_dir).unwrap_or(0);

                println!(
                    "Found {} templates ({} custom loaded):",
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
                        t.template_id, t.domain, t.name, parent
                    );
                }
            }
            TemplateCommands::New {
                name,
                domain,
                extends,
            } => {
                let yaml = TemplateRegistry::create_custom_starter(
                    &name,
                    &domain,
                    extends.as_deref(),
                )?;
                let file_name = format!("{}.yaml", name.to_lowercase().replace(' ', "-"));
                let out_dir = Path::new("templates");
                std::fs::create_dir_all(out_dir)?;
                let out_path = out_dir.join(&file_name);
                std::fs::write(&out_path, &yaml)?;
                println!(
                    "Created custom template starter at: {}",
                    out_path.display()
                );
            }
            TemplateCommands::Check { path } => {
                let content = std::fs::read_to_string(&path)?;
                match Template::from_yaml(&content) {
                    Ok(template) => {
                        println!("Template '{}' is VALID:", template.template_id);
                        println!("  Name: {}", template.name);
                        println!("  Domain: {}", template.domain);
                        println!("  Parameters defined: {}", template.parameters.len());
                        println!("  Locked fields: {}", template.locked.len());
                    }
                    Err(e) => {
                        eprintln!("Template validation FAILED: {e}");
                        std::process::exit(1);
                    }
                }
            }
            TemplateCommands::SaveRun { run_id, name } => {
                println!("Saving run {run_id} as template '{name}'...");
            }
        },
        Some(Commands::Export { format, template, output }) => {
            println!("Exporting template '{template}' as pure standalone {format} to: {output}");
            let mut registry = TemplateRegistry::new();
            let _ = registry.load_from_dir(Path::new("templates"));

            let tmpl = registry.get(&template).ok_or_else(|| {
                anyhow::anyhow!("Template '{}' not found in registry", template)
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
            println!("  sentinel template list");
            println!("  sentinel dev");
        }
        None => {
            // Default run on port 2512
            run_dev(2512).await?;
        }
    }

    Ok(())
}

async fn run_dev(port: u16) -> anyhow::Result<()> {
    let supervisor = sentinel_daemon::supervisor::ProcessSupervisor::new();
    let gpu_pool = sentinel_daemon::scheduler::DynamicGpuPool::new(4);
    let hardware_monitor = std::sync::Arc::new(tokio::sync::Mutex::new(
        sentinel_daemon::scheduler::HardwareMonitor::new(),
    ));
    let mut registry = TemplateRegistry::new();
    let _ = registry.load_from_dir(Path::new("templates"));
    let templates = std::sync::Arc::new(registry);

    let state = sentinel_daemon::server::AppState {
        supervisor,
        gpu_pool,
        hardware_monitor,
        templates,
    };

    let app = sentinel_daemon::server::Server::router(state);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("🚀 Sentinel Daemon active at http://localhost:{port}");
    println!("   API health: http://localhost:{port}/api/health");
    println!("   WebSocket telemetry: ws://localhost:{port}/ws/live");
    axum::serve(listener, app).await?;
    Ok(())
}
