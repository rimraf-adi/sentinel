//! The React dashboard, compiled by Vite (`ui/`) and embedded into the binary.
//! Debug builds read `ui/dist` from disk, so rebuilding the UI needs no Rust recompile.

use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "../../ui/dist/"]
struct Assets;

fn serve(path: &str) -> Option<Response> {
    let file = Assets::get(path)?;
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    Some(
        (
            [
                (header::CONTENT_TYPE, mime_guess::from_path(path).first_or_octet_stream().to_string()),
                (header::CACHE_CONTROL, cache.to_string()),
            ],
            file.data,
        )
            .into_response(),
    )
}

/// Static files, with SPA fallback to `index.html` for client-side routes.
pub async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.starts_with("api/") || path.starts_with("ws/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    // Old clients and crawlers ask for /favicon.ico; answer with the SVG instead of the SPA's index.html.
    let path = match path {
        "" => "index.html",
        "favicon.ico" => "favicon.svg",
        p => p,
    };
    serve(path)
        .or_else(|| serve("index.html"))
        .unwrap_or_else(|| (StatusCode::NOT_FOUND, "UI assets not embedded").into_response())
}
