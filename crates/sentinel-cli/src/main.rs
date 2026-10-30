use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "sentinel", author, version, about = "Sentinel ML Training Supervisor & Framework", long_about = None)]
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
    New { name: String },
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
            TemplateCommands::List => println!("Listing templates..."),
            TemplateCommands::New { name } => println!("Creating template: {name}"),
            TemplateCommands::Check { path } => println!("Validating template: {path}"),
            TemplateCommands::SaveRun { run_id, name } => {
                println!("Saving run {run_id} as template {name}")
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
