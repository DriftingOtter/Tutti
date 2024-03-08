use std::io::{self, Write};
use std::fs::File;
use std::io::BufReader;
use rodio::{Decoder, OutputStream, source::Source};

fn play_audio(file_path: String, song_duration: u64) -> bool {
    // Get a output stream handle to the default physical sound device
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();

    let file = BufReader::new(File::open(file_path).unwrap());
    let source = Decoder::new(file).unwrap();

    stream_handle.play_raw(source.convert_samples())
        .expect("Error during playing audio.");

    // keep thread alive until end of song.
    std::thread::sleep(std::time::Duration::from_secs(song_duration));

    return true;
}

fn get_song_name(file_path: String) -> String {
    let metadata = metadata::media_file::MediaFileMetadata::new(&std::path::Path::new(&file_path));
    let song_title = Some(metadata.unwrap().title).take().flatten().unwrap();

    return song_title;
}

fn get_song_duration(file_path: String) -> u64 {
    let song_duration = metadata::media_file::MediaFileMetadata::new(&std::path::Path::new(&file_path))
        .unwrap()
        ._duration;
    
    match song_duration {
        Some(duration) => {
            if duration < 0.0 {
                return 0;
            } else {
                return duration as u64;
            }
        },
        None => 0,
    }
}

fn main() {

    println!("-+-+-+-+-+-+-+-+-+-");
    println!("Tutti: Music Player");
    println!("-+-+-+-+-+-+-+-+-+-");

    loop{

        let mut song_path: String = String::new();

        print!("Song Path > ");
        io::stdout().flush().expect("Could not flush terminal.");

        io::stdin()
            .read_line(&mut song_path)
            .expect("Error during I/O reading");

        song_path = song_path.trim().parse().expect("Error during path parsing.");

        let song_name: String = get_song_name(song_path.clone());
        let song_duration: u64 = get_song_duration(song_path.clone());

        println!("Playing: {}", song_name);
        play_audio(song_path, song_duration);

    }
}
