mod pitch;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

const SERVER_URL: &str = "http://127.0.0.1:8572";

fn log(msg: &str) {
    let log_path = std::env::current_exe()
        .map(|p| p.with_file_name("ccbsampler-client.log"))
        .unwrap_or_else(|_| "ccbsampler-client.log".into());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(f, "{msg}");
    }
}

#[derive(Parser, Debug)]
#[command(version)]
#[command(allow_negative_numbers = true)]
struct Args {
    in_file: String,
    out_file: String,
    pitch: String,
    velocity: f64,
    #[arg(default_value = "")]
    flags: String,
    #[arg(default_value_t = 0.0)]
    offset: f64,
    #[arg(default_value_t = 1000)]
    length: i64,
    #[arg(default_value_t = 0.0)]
    consonant: f64,
    #[arg(default_value_t = 0.0)]
    cutoff: f64,
    #[arg(default_value_t = 100.0)]
    volume: f64,
    #[arg(default_value_t = 0.0)]
    modulation: f64,
    #[arg(default_value = "!100")]
    tempo: String,
    #[arg(default_value = "AA")]
    pitch_string: String,
}

#[derive(Serialize)]
struct ResampleRequest {
    in_file: String,
    out_file: String,
    pitch: String,
    velocity: f64,
    flags: HashMap<String, serde_json::Value>,
    offset: f64,
    length: i64,
    consonant: f64,
    cutoff: f64,
    volume: f64,
    modulation: f64,
    tempo: f64,
    pitchbend: Vec<f64>,
}

#[derive(Deserialize, Debug)]
struct RenderMetrics {
    feature_ms: f64,
    queue_ms: f64,
    infer_ms: f64,
    total_ms: f64,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ResampleResponse {
    status: String,
    out_file: Option<String>,
    error: Option<String>,
    metrics: Option<RenderMetrics>,
}

fn main() -> Result<()> {
    let raw_args: Vec<String> = std::env::args().collect();
    log(&format!("=== args: {raw_args:?}"));

    let args = Args::parse();

    let tempo: f64 = args
        .tempo
        .trim_start_matches('!')
        .parse()
        .context("Invalid tempo")?;

    let pitchbend = pitch::decode(&args.pitch_string);
    let flags = pitch::parse_flags(&args.flags);

    let req = ResampleRequest {
        in_file: args.in_file,
        out_file: args.out_file,
        pitch: args.pitch,
        velocity: args.velocity,
        flags,
        offset: args.offset,
        length: args.length,
        consonant: args.consonant,
        cutoff: args.cutoff,
        volume: args.volume,
        modulation: args.modulation,
        tempo,
        pitchbend,
    };

    let json = serde_json::to_string_pretty(&req).unwrap_or_default();
    log(&format!("=== request:\n{json}"));

    log("=== sending request...");
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .context("Failed to build HTTP client")?;
    let send_result = client
        .post(format!("{SERVER_URL}/resample"))
        .json(&req)
        .send();

    let http_resp = match send_result {
        Ok(r) => r,
        Err(e) => {
            log(&format!("=== send error: {e}"));
            anyhow::bail!("Failed to connect to ccbsampler server: {e}");
        }
    };

    log(&format!("=== http status: {}", http_resp.status()));
    let resp: ResampleResponse = match http_resp.json() {
        Ok(r) => r,
        Err(e) => {
            log(&format!("=== json parse error: {e}"));
            anyhow::bail!("Failed to parse server response: {e}");
        }
    };

    log(&format!("=== response: status={}", resp.status));

    if resp.status != "ok" {
        let err = resp.error.unwrap_or_else(|| "(no error message)".into());
        log(&format!("=== error: {err}"));
        anyhow::bail!("Server error: {err}");
    }

    if let Some(m) = resp.metrics {
        let metrics = format!(
            "feature={:.0}ms  queue={:.0}ms  infer={:.0}ms  total={:.0}ms",
            m.feature_ms, m.queue_ms, m.infer_ms, m.total_ms
        );
        log(&format!("=== metrics: {metrics}"));
        eprintln!("{metrics}");
    }

    Ok(())
}
