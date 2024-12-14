use image::ImageReader;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let original_img = ImageReader::open("Castle5565.jpg")?.decode()?;
    seam_carving_rust::execute(original_img, 100);
    Ok(())
}
