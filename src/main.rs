use audiotags::{Error, Tag};
use regex::Regex;
use rodio::{Decoder, OutputStream, Sink};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Write};
use std::ops::Add;
use std::thread;

pub mod parser;

static CACHE_FILE_PATH: &str = "cache.json";

fn play_audio(sound_source: String) -> thread::JoinHandle<()> {
    let higgs = thread::spawn(move || {
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();

        let file = BufReader::new(File::open(sound_source).unwrap());
        let source = Decoder::new(file).unwrap();
        sink.append(source);
        sink.sleep_until_end();
    });

    return higgs;
}
fn get_song_metadata(
    file_path: &str,
) -> Result<(String, String, String, String, String, u16, u16, f64), Error> {
    let metadata = Tag::default().read_from_path(file_path)?;
    let song_data = (
        metadata.title().unwrap_or("").to_string(),
        metadata.album_title().unwrap_or("").to_string(),
        metadata.year().map(|y| y.to_string()).unwrap_or_default(),
        metadata.genre().unwrap_or("").to_string(),
        metadata.album_artist().unwrap_or("").to_string(),
        metadata.track_number().unwrap_or(0),
        metadata.total_tracks().unwrap_or(0),
        metadata.duration().unwrap_or(0.0),
    );

    Ok(song_data)
}

fn print_dir() -> io::Result<()> {
    for entry in fs::read_dir("/home/daksh/Music").unwrap() {
        let dir = entry.unwrap();
        let song_name = extract_file_name(dir.file_name().to_str().unwrap()).unwrap();
        println!("{}", song_name);
    }
    Ok(())
}

fn extract_file_name(file_name: &str) -> io::Result<String> {
    let re = Regex::new(r"\.(mp3|flac|wav|ogg)$").unwrap();
    let file_name_without_extension = match re.captures(file_name) {
        Some(captures) => {
            let file_name_start = 0;
            let file_name_end = captures.get(0).unwrap().start();
            &file_name[file_name_start..file_name_end]
        }
        None => file_name,
    };

    Ok(file_name_without_extension.to_string())
}

fn format_song_length(mut minutes: f64, mut seconds: f64) -> Vec<String> {
    if seconds >= 60.0 {
        let additional_minutes = (seconds / 60.0).floor();
        minutes += additional_minutes;
        seconds -= additional_minutes * 60.0;
    }

    let mut song_min_str = minutes.to_string();
    let mut song_sec_str = seconds.to_string();

    if minutes < 10.0 {
        song_min_str.insert(0, '0');
    }

    if seconds < 10.0 {
        song_sec_str.insert(0, '0');
    }

    return vec![song_min_str, song_sec_str];
}

fn find_root_dir() -> String {
    let root_dir: String = match env::var("HOME") {
        Ok(val) => val.add("/Music/"),
        Err(_) => {
            println!("Unable to determine home directory. Exiting.");
            return "".to_string();
        }
    };

    return root_dir;
}

fn get_user_input(root_directory: String) -> Vec<String> {
    print_dir().expect("Could not print directory information.");

    let mut song_path: String = root_directory.clone();
    let mut song_name: String = String::new();

    print!("Song Path > ");
    io::stdout().flush().expect("Could not flush terminal.");

    io::stdin()
        .read_line(&mut song_name)
        .expect("Error during I/O reading");

    song_path = song_path
        .add(song_name.as_str())
        .trim()
        .parse()
        .expect("Error during path parsing.");

    song_path = song_path.add(".mp3");

    return vec![song_name, song_path];
}

fn display_metadata(song_path: String) {
    let song_metadata = get_song_metadata(song_path.clone().as_str());

    println!(
        "Playing: {song_title} by {artists} on {album} {year}",
        song_title = &song_metadata.as_ref().unwrap().0,
        artists    = &song_metadata.as_ref().unwrap().4,
        album      = &song_metadata.as_ref().unwrap().1,
        year       = &song_metadata.as_ref().unwrap().2
    );

    let song_min = &song_metadata.as_ref().unwrap().7 / 60.0;
    let song_sec = &song_metadata.as_ref().unwrap().7 % 60.0;

    let song_duration_formatted = format_song_length(song_min, song_sec);

    println!(
        "Duration: {minutes}:{seconds}",
        minutes = &song_duration_formatted[0],
        seconds = &song_duration_formatted[1]
    );
}

fn get_similar_song(song_name: String) {
    let cache_content = parser::get_cache(CACHE_FILE_PATH).unwrap_or_else(|err| {
        eprintln!("Error reading cache file: {}", err);
        std::process::exit(1);
    });

    let target_song_data = parser::get_song_cache_data(&cache_content, song_name.clone().as_str())
        .unwrap_or_else(|| {
            eprintln!("Played song not found in cache.");
            std::process::exit(1);
        });

    match parser::get_similar_song_cache_data(
        &cache_content,
        target_song_data.happy,
        target_song_data.sad,
        target_song_data.angry,
        song_name.clone().as_str(),
    ) {
        Some(song) => {
            println!("Found next closest song to '{}':", song_name);
            println!("Name: {}", song.name);
            println!("Happy: {}", song.happy);
            println!("Sad: {}", song.sad);
            println!("Angry: {}", song.angry);
        }
        None => println!("No similar song found."),
    }
}

fn main() {

    // Find Root Dir & Getting Usr Input
    let root_directory: String = find_root_dir();
    let song_dictionary = get_user_input(root_directory);

    let song_name: String = song_dictionary[0].clone();
    let song_path: String = song_dictionary[1].clone();

    println!("{}", &song_path);


    // Gathering & Displaying Current Song Metadata
    let metadata_path: String = song_path.clone();
    display_metadata(metadata_path);


    // Playing Song
    let audio_source = play_audio(song_path.clone());
    audio_source.join().unwrap();


    // Get Similar Song HSA Values
    get_similar_song(song_name);

}
