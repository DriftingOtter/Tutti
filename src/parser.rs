use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
pub struct EmotionCluster {
    pub name: String,
    pub happy: f64,
    pub sad: f64,
    pub angry: f64,
}

pub fn get_cache(cache_path: &str) -> std::io::Result<String> {
    fs::read_to_string(cache_path)
}

pub fn get_song_cache_data(cache: &str, record: &str) -> Option<EmotionCluster> {
    let parsed_song_cache: Value = serde_json::from_str(cache).ok()?;

    if let Some(song_data) = parsed_song_cache.get(record) {
        let emotion_cluster = EmotionCluster {
            name: record.to_string(),
            happy: song_data["happy"].as_f64().unwrap_or_default(),
            sad: song_data["sad"].as_f64().unwrap_or_default(),
            angry: song_data["angry"].as_f64().unwrap_or_default(),
        };
        Some(emotion_cluster)
    } else {
        None
    }
}

pub fn get_similar_song_cache_data(
    cache: &str,
    target_happy: f64,
    target_sad: f64,
    target_angry: f64,
    current_song: &str,
) -> Option<EmotionCluster> {
    let parsed_song_cache: Value = serde_json::from_str(cache).ok()?;
    let mut closest_song: Option<EmotionCluster> = None;
    let mut min_distance = std::f64::INFINITY;

    for (song_name, song_data) in parsed_song_cache.as_object().unwrap().iter() {
        if song_name == current_song {
            continue; // Skip current song
        }

        let happy = song_data["happy"].as_f64().unwrap_or_default();
        let sad = song_data["sad"].as_f64().unwrap_or_default();
        let angry = song_data["angry"].as_f64().unwrap_or_default();
        let distance =
            (target_happy - happy).abs() + (target_sad - sad).abs() + (target_angry - angry).abs();

        if distance < min_distance {
            min_distance = distance;
            closest_song = Some(EmotionCluster {
                name: song_name.to_string(),
                happy,
                sad,
                angry,
            });
        }
    }

    closest_song
}
