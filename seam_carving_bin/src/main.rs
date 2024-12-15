use bpaf::Bpaf;
use seam_carving_lib::open_image;
use std::{error::Error, ffi::OsString, path::PathBuf};

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options)]
pub struct CliOptions {
    /// Path for the image that will be carved.
    image_path: OsString,
    /// Number of columsns to be carved.
    n_cols: u32,
    /// Path for the output image.
    out_image: Option<OsString>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let image_path = PathBuf::from("Castle5565.jpg");
    let original_img = open_image(image_path.as_path())?;
    seam_carving_lib::execute(original_img, 100);
    Ok(())
}
