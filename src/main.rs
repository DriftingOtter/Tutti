use audiotags::Tag;
use rodio::{source::Source, Decoder, OutputStream};
use std::fs::File;
use std::io::BufReader;
use std::io::{self, Write};

fn play_audio(file_path: String, song_duration: u64) -> bool {
    // Get a output stream handle to the default physical sound device
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();

    let file = BufReader::new(File::open(file_path).unwrap());
    let source = Decoder::new(file).unwrap();

    stream_handle
        .play_raw(source.convert_samples())
        .expect("Error during playing audio.");

    // keep thread alive until end of song.
    std::thread::sleep(std::time::Duration::from_secs(song_duration));

    return true;
}

fn get_song_metadata(
    file_path: String,
) -> (
    String,
    String,
    String,
    String,
    String,
    u16,
    u16,
) {
    let metadata = Tag::new().read_from_path(&file_path).unwrap();
    let mut song_data: (
        String,
        String,
        String,
        String,
        String,
        u16,
        u16,
    ) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        0,
        0,
    );

    song_data.0 = metadata.title().unwrap_or("").to_string();
    song_data.1 = metadata.album_title().unwrap_or("").to_string();
    song_data.2 = metadata.year().map(|y| y.to_string()).unwrap_or_default();
    song_data.3 = metadata.genre().unwrap_or("").to_string();
    song_data.4 = metadata.album_artist().unwrap_or("").to_string();
    song_data.5 = metadata.track_number().unwrap_or(0);
    song_data.6 = metadata.total_tracks().unwrap_or(0);

    return song_data;
}

fn get_song_duration(file_path: String) -> u64 {
    let song_duration = Tag::new().read_from_path(&file_path).unwrap().duration();
    match song_duration {
        Some(duration) => {
            if duration < 0.0 {
                return 0;
            } else {
                return duration as u64;
            }
        }
        None => 0,
    }
}

fn main() {

    println!("-+-+-+-+-+-+-+-+-+-");
    println!("Tutti: Music Player");
    println!("-+-+-+-+-+-+-+-+-+-");

    loop {
        let mut song_path: String = String::new();

        print!("Song Path > ");
        io::stdout().flush().expect("Could not flush terminal.");

        io::stdin()
            .read_line(&mut song_path)
            .expect("Error during I/O reading");

        song_path = song_path
            .trim()
            .parse()
            .expect("Error during path parsing.");

        let song_metadata = get_song_metadata(song_path.clone());
        let song_duration = get_song_duration(song_path.clone());

        println!(
            "Playing: {song_title} by {artists} on {album} {year}",
            song_title = song_metadata.0,
            artists = song_metadata.4,
            album = song_metadata.1,
            year = song_metadata.2
        );
        println!("Duration: {}", song_duration / 60);

        play_audio(song_path, song_duration);
    }
}
