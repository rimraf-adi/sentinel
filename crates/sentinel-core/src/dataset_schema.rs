//! Inspect a CSV so the UI can propose features, target, task and loss instead of hardcoding them.

use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

const DISTINCT_CAP: usize = 100;
const CLASS_LIMIT: usize = 20;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Column {
    pub name: String,
    /// numeric | integer | categorical
    pub kind: String,
    /// Distinct non-empty values, capped at 100 (see `distinct_capped`).
    pub distinct: usize,
    pub distinct_capped: bool,
    pub missing: usize,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Suggestion {
    pub target: String,
    pub features: Vec<String>,
    /// classification | regression
    pub task: String,
    pub loss: String,
    /// Number of classes (classification) or 1 (regression): the size of the output layer.
    pub outputs: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CsvSchema {
    pub rows: usize,
    pub columns: Vec<Column>,
    pub suggestion: Option<Suggestion>,
}

const TARGET_NAMES: &[&str] = &["label", "target", "y", "class", "species", "outcome"];

struct Acc {
    name: String,
    numeric: bool,
    integer: bool,
    seen: BTreeSet<String>,
    capped: bool,
    missing: usize,
}

pub fn inspect_csv(path: &Path) -> Result<CsvSchema, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let headers: Vec<String> = rdr.headers().map_err(|e| e.to_string())?.iter().map(str::to_string).collect();
    if headers.is_empty() {
        return Err("file has no header row".into());
    }
    let mut cols: Vec<Acc> = headers
        .iter()
        .map(|h| Acc { name: h.clone(), numeric: true, integer: true, seen: BTreeSet::new(), capped: false, missing: 0 })
        .collect();

    let mut rows = 0usize;
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("row {}: {e}", rows + 2))?;
        rows += 1;
        for (i, c) in cols.iter_mut().enumerate() {
            let v = rec.get(i).unwrap_or("");
            if v.is_empty() {
                c.missing += 1;
                continue;
            }
            if c.numeric && v.parse::<f64>().map_or(true, |x| !x.is_finite()) {
                c.numeric = false;
            }
            if c.integer && v.parse::<i64>().is_err() {
                c.integer = false;
            }
            if c.seen.len() < DISTINCT_CAP {
                c.seen.insert(v.to_string());
            } else if !c.seen.contains(v) {
                c.capped = true;
            }
        }
    }
    if rows == 0 {
        return Err("file has a header but no data rows".into());
    }

    let columns: Vec<Column> = cols
        .into_iter()
        .map(|c| Column {
            kind: if c.integer && c.numeric { "integer" } else if c.numeric { "numeric" } else { "categorical" }.into(),
            distinct: c.seen.len(),
            distinct_capped: c.capped,
            missing: c.missing,
            examples: c.seen.iter().take(4).cloned().collect(),
            name: c.name,
        })
        .collect();
    let suggestion = suggest(&columns);
    Ok(CsvSchema { rows, columns, suggestion })
}

fn suggest(cols: &[Column]) -> Option<Suggestion> {
    let target = cols
        .iter()
        .find(|c| TARGET_NAMES.contains(&c.name.to_lowercase().as_str()))
        .or_else(|| cols.last())?;
    let classification = target.kind == "categorical" || (target.kind == "integer" && target.distinct <= CLASS_LIMIT && !target.distinct_capped);
    let features: Vec<String> = cols
        .iter()
        .filter(|c| c.name != target.name && (c.kind == "numeric" || c.kind == "integer"))
        .map(|c| c.name.clone())
        .collect();
    if features.is_empty() || (classification && target.distinct_capped) {
        return None;
    }
    Some(Suggestion {
        target: target.name.clone(),
        features,
        task: if classification { "classification" } else { "regression" }.into(),
        loss: if classification { "cross_entropy" } else { "mse" }.into(),
        outputs: if classification { target.distinct } else { 1 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv_file(tag: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("sentinel-ds-{tag}-{}.csv", std::process::id()));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn string_labels_mean_classification_with_class_count() {
        let p = csv_file("cls", "sepal,petal,species\n5.1,1.4,setosa\n6.0,4.5,versicolor\n6.9,5.8,virginica\n5.0,1.5,setosa\n");
        let s = inspect_csv(&p).unwrap();
        assert_eq!(s.rows, 4);
        let sg = s.suggestion.unwrap();
        assert_eq!((sg.target.as_str(), sg.task.as_str(), sg.loss.as_str(), sg.outputs), ("species", "classification", "cross_entropy", 3));
        assert_eq!(sg.features, ["sepal", "petal"]);
    }

    #[test]
    fn continuous_target_means_regression() {
        let p = csv_file("reg", "a,b,y\n1,2,3.5\n2,1,4.25\n3,5,9.0\n4,4,8.5\n");
        let sg = inspect_csv(&p).unwrap().suggestion.unwrap();
        assert_eq!((sg.task.as_str(), sg.loss.as_str(), sg.outputs), ("regression", "mse", 1));
    }

    #[test]
    fn low_cardinality_integer_target_is_classification() {
        let p = csv_file("int", "x1,x2,label\n0.1,0.2,0\n0.3,0.1,1\n0.9,0.8,2\n0.5,0.5,1\n");
        let sg = inspect_csv(&p).unwrap().suggestion.unwrap();
        assert_eq!((sg.task.as_str(), sg.outputs), ("classification", 3));
    }

    #[test]
    fn target_is_found_by_name_not_only_by_position() {
        let p = csv_file("name", "label,f1,f2\na,1,2\nb,2,3\na,3,4\n");
        let sg = inspect_csv(&p).unwrap().suggestion.unwrap();
        assert_eq!(sg.target, "label");
        assert_eq!(sg.features, ["f1", "f2"]);
    }

    #[test]
    fn counts_missing_and_quoted_commas() {
        let p = csv_file("miss", "name,v,y\n\"Smith, J\",1,0\nDoe,,1\n");
        let s = inspect_csv(&p).unwrap();
        assert_eq!(s.columns[1].missing, 1);
        assert_eq!(s.columns[0].examples.len(), 2);
    }

    #[test]
    fn errors_are_readable() {
        assert!(inspect_csv(Path::new("/no/such.csv")).unwrap_err().contains("cannot read"));
        assert!(inspect_csv(&csv_file("hdr", "a,b\n")).unwrap_err().contains("no data rows"));
        // all-text file: nothing numeric to learn from
        assert!(inspect_csv(&csv_file("txt", "a,b\nx,y\nz,w\n")).unwrap().suggestion.is_none());
    }
}
