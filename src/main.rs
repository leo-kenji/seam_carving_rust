use image::{io::Reader as ImageReader, DynamicImage, GrayImage, ImageBuffer, Luma, Rgb};
use ndarray::{s, Array2, Axis};
use num_traits::Zero;

use std::{error::Error, ops::AddAssign, vec};

fn apply_sobel<T: image::Primitive>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
) -> ImageBuffer<Luma<T>, Vec<T>> {
    let w = img.width();
    let h = img.height();

    let mut buffer = Vec::with_capacity((w * h) as usize);
    buffer.resize((w * h) as usize, 0.);

    // TODO check if the image has the minimum size
    for i in 0..w {
        for j in 0..h {
            let i1 = match i {
                0 => w - 1,
                _ => i - 1,
            };
            let i2 = i;
            let i3 = match i {
                x if x == w - 1 => 0,
                _ => i + 1,
            };

            let j1 = match j {
                0 => h - 1,
                _ => j - 1,
            };
            let j2 = j;
            let j3 = match j {
                x if x == h - 1 => 0,
                _ => j + 1,
            };

            let val1 = img.get_pixel(i1, j1).0[0].to_f32().unwrap();
            let val2 = img.get_pixel(i2, j1).0[0].to_f32().unwrap();
            let val3 = img.get_pixel(i3, j1).0[0].to_f32().unwrap();

            let val4 = img.get_pixel(i1, j2).0[0].to_f32().unwrap();
            // let val5 = img.get_pixel(i2, j2).0[0].to_f32().unwrap();
            let val6 = img.get_pixel(i3, j2).0[0].to_f32().unwrap();

            let val7 = img.get_pixel(i1, j3).0[0].to_f32().unwrap();
            let val8 = img.get_pixel(i2, j3).0[0].to_f32().unwrap();
            let val9 = img.get_pixel(i3, j3).0[0].to_f32().unwrap();

            let s_x = val1 - val3 + 2. * val4 - 2. * val6 + val7 - val8;
            let s_y = val1 + 2. * val2 + val3 - val7 - 2. * val8 - val9;

            let mag = (s_x * s_x + s_y * s_y).sqrt();

            buffer[(j * w + i) as usize] = mag;
        }
    }
    // Normalize
    let max = buffer.iter().fold(f32::NEG_INFINITY, |max, &x| x.max(max));

    // TODO Make this in place?
    let buffer = buffer
        .iter()
        .map(|x| x / max * 255.)
        .map(|x| T::from(x).unwrap())
        .collect::<Vec<T>>();

    ImageBuffer::from_raw(w, h, buffer).unwrap()
}

fn compute_energy(img: GrayImage) -> GrayImage {
    apply_sobel(&img)
}
// TODO test crate nshare

fn compute_minimum_energy_map<T, U>(energy: &Array2<T>) -> Array2<U>
where
    T: Clone,
    U: Zero + From<T> + PartialOrd + AddAssign + Copy,
{
    let w = energy.ncols();
    let h = energy.nrows();

    let mut min_energy: Array2<U> = energy.mapv(|x| U::from(x));

    for row in 1..h {
        // TODO maybe this internal loop can be parallelized
        for column in 0..w {
            let min_c = column.saturating_sub(1);
            let max_c = w.min(column + 2);

            let min_path = *(min_energy
                .slice(s![row - 1, min_c..max_c])
                .iter()
                .min_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap());
            min_energy[[row, column]] += min_path;
        }
    }

    min_energy
}

fn find_min_energy_path(energy: &GrayImage) -> Vec<usize> {
    let n_rows = energy.height();
    let w = energy.width();

    // TODO This probably does not need to be mut
    let mut energy = Array2::from_shape_vec(
        (energy.height() as usize, energy.width() as usize),
        energy.as_raw().clone(),
    )
    .unwrap();

    energy.invert_axis(Axis(0));
    let mut idx = vec![0; n_rows as usize];
    idx[0] = energy
        .index_axis(Axis(0), 1)
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| a.cmp(&b))
        .map(|(index, _)| index)
        .unwrap();
    // for (i, row) in energy.rows().enumerate()
    for (i, row) in energy.axis_iter(Axis(0)).enumerate().skip(1) {
        let last_i = idx[i - 1];
        let min_i = last_i.saturating_sub(1);
        let max_i = w.min(last_i as u32 + 2) as usize;

        let temp = row
            .slice(s![min_i..max_i])
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.cmp(&b))
            .map(|(index, _)| index)
            .unwrap();
        idx[i] = temp + min_i;
    }

    idx.iter().rev().cloned().collect()
}

fn remove_path(img: &DynamicImage, path: Vec<usize>) -> DynamicImage {
    let img = img.clone().into_rgb16();
    let width = img.width();
    let height = img.height();

    let mut out: ImageBuffer<Rgb<u16>, _> = ImageBuffer::new(width - 1, height);

    for (y, row) in out.rows_mut().enumerate() {
        for (x, pixel) in row.enumerate() {
            let z = if x >= path[y] { x + 1 } else { x };
            *pixel = *img.get_pixel(z as u32, y as u32);
        }
    }
    out.into()
}

fn main() -> Result<(), Box<dyn Error>> {
    let original_img = ImageReader::open("Castle5565.jpg")?.decode()?;

    let mut image: DynamicImage = original_img;
    for _ in 0..400 {
        let energy = compute_energy(image.to_luma8());
        let min_energy = compute_minimum_energy_map(&energy);
        let path = find_min_energy_path(&min_energy);
        image = remove_path(&image, path);
    }
    image.save("output.png").unwrap();

    Ok(())
}
