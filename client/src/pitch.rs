use serde_json::Value;
use std::collections::HashMap;

fn to_uint6(c: char) -> i32 {
    let c = c as i32;
    match c {
        97..=122 => c - 71, // a-z
        65..=90 => c - 65,  // A-Z
        48..=57 => c + 4,   // 0-9
        43 => 62,           // +
        47 => 63,           // /
        _ => 0,
    }
}

fn to_int12(s: &str) -> i32 {
    let mut chars = s.chars();
    let hi = to_uint6(chars.next().unwrap_or('A'));
    let lo = to_uint6(chars.next().unwrap_or('A'));
    let uint12 = (hi << 6) | lo;
    if (uint12 >> 11) & 1 == 1 {
        uint12 - 4096
    } else {
        uint12
    }
}

fn decode_b64_stream(s: &str) -> Vec<f64> {
    (0..s.len())
        .step_by(2)
        .filter(|&i| i + 1 < s.len())
        .map(|i| to_int12(&s[i..i + 2]) as f64)
        .collect()
}

pub fn decode(pitch_string: &str) -> Vec<f64> {
    if pitch_string.is_empty() || pitch_string == "AA" {
        return vec![0.0];
    }

    let parts: Vec<&str> = pitch_string.split('#').collect();
    let mut result: Vec<f64> = Vec::new();

    let mut i = 0;
    while i < parts.len() {
        let decoded = decode_b64_stream(parts[i]);
        result.extend_from_slice(&decoded);
        i += 1;
        if i < parts.len() {
            if let Ok(rle) = parts[i].parse::<usize>() {
                if let Some(&last) = result.last() {
                    result.extend(std::iter::repeat(last).take(rle));
                }
            }
            i += 1;
        }
    }

    if result.iter().all(|&v| v == result[0]) {
        return vec![0.0; result.len()];
    }

    result
}

pub fn parse_flags(flags: &str) -> HashMap<String, Value> {
    const FLAG_KEYS: &[&str] = &[
        "Hb", "Hv", "Ht", "He", "fe", "fl", "fo", "fv", "fp", "ve", "vo", "g", "t", "A", "B", "P",
        "S", "p", "R", "D", "C", "Z", "G",
    ];

    let mut map = HashMap::new();
    let mut s = flags.trim_start_matches('/');

    while !s.is_empty() {
        let matched = FLAG_KEYS.iter().find(|&&k| s.starts_with(k));
        let Some(&key) = matched else {
            s = &s[s.char_indices().nth(1).map(|(i, _)| i).unwrap_or(s.len())..];
            continue;
        };

        s = &s[key.len()..];

        let num_end = s
            .char_indices()
            .take_while(|(i, c)| (*i == 0 && (*c == '+' || *c == '-')) || c.is_ascii_digit())
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);

        let value: Value = if num_end > 0 {
            let num: i64 = s[..num_end].parse().unwrap_or(0);
            s = &s[num_end..];
            Value::Number(num.into())
        } else {
            Value::Bool(true)
        };

        map.insert(key.to_string(), value);
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_flat() {
        let result = decode("AA");
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_decode_rle() {
        let result = decode("AA#3");
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_parse_flags_numeric() {
        let flags = parse_flags("g-50Hb80");
        assert_eq!(flags["g"], Value::Number((-50_i64).into()));
        assert_eq!(flags["Hb"], Value::Number(80_i64.into()));
    }

    #[test]
    fn test_parse_flags_switch() {
        let flags = parse_flags("HeG");
        assert_eq!(flags["He"], Value::Bool(true));
        assert_eq!(flags["G"], Value::Bool(true));
    }

    #[test]
    fn test_parse_flags_mixed() {
        let flags = parse_flags("/g+10He");
        assert_eq!(flags["g"], Value::Number(10_i64.into()));
        assert_eq!(flags["He"], Value::Bool(true));
    }
}
