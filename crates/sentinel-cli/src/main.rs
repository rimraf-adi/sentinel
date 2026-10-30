use clap::{Parser, Subcommand};
use sentinel_core::templates::{Template, TemplateRegistry};
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
        #[arg(short, long, default_value = "8080")]
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
            println!("Starting Sentinel daemon on port {port}...");
        }
        Some(Commands::Run { config }) => {
            println!("Launching Sentinel run with config: {config}");
        }
        Some(Commands::Sweep { config }) => {
            println!("Launching Sentinel sweep with config: {config}");
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
        Some(Commands::Export { format }) => {
            println!("Exporting to {format}...");
        }
        Some(Commands::New { name }) => {
            println!("Scaffolding new Sentinel project: {name}");
        }
        None => {
            println!("Sentinel v0.1.0 — run 'sentinel --help' for commands");
        }
    }

    Ok(())
}
