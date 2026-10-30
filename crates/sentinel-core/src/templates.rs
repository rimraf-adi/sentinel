//! Template engine interfaces.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub domain: String,
}
