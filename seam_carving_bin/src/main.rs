// // use bpaf::Bpaf;
use bpaf::*;
use seam_carving_lib::open_image;
use std::{error::Error, path::PathBuf};

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options)]
pub struct CliOptions {
    /// Path for the output image.
    out_image: Option<PathBuf>,

    /// Path for the image that will be carved.
    #[bpaf(positional("IMAGE PATH"), fallback(PathBuf::from("Castle5565.jpg")))]
    image_path: PathBuf,

    /// Number of columns to be carved.
    #[bpaf(positional("NUMBER OF COLUMNS TO BE REMOVED"), fallback(10))]
    n_cols: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli_options = cli_options().run();
    let image_path = cli_options.image_path;
    let original_img = open_image(image_path.as_path())?;
    seam_carving_lib::execute(original_img, cli_options.n_cols);
    Ok(())
}
