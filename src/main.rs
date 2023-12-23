use image::{io::Reader as ImageReader, DynamicImage, GrayImage, ImageBuffer, Rgb};
use ndarray::{s, Array2, Axis, Zip};
use num_traits::{real::Real, Num, ToPrimitive, Zero};
use std::{error::Error, ops::AddAssign, vec};

fn apply_sobel<T, U>(img: &Array2<T>) -> Array2<U>
where
    T: Num + ToPrimitive + Copy + Sync,
    U: Clone + Zero + Real + Send + Zero,
{
    let w = img.ncols();
    let h = img.nrows();

    let mut buffer = Array2::zeros((h, w));

    Zip::indexed(&mut buffer).for_each(|(j, i), x| {
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

        let val1 = U::from(img[[j1, i1]]).unwrap();
        let val2 = U::from(img[[j1, i2]]).unwrap();
        let val3 = U::from(img[[j1, i3]]).unwrap();

        let val4 = U::from(img[[j2, i1]]).unwrap();
        // let val5 = U::from(img[[j2, i2]]).unwrap();
        let val6 = U::from(img[[j2, i3]]).unwrap();

        let val7 = U::from(img[[j3, i1]]).unwrap();
        let val8 = U::from(img[[j3, i2]]).unwrap();
        let val9 = U::from(img[[j3, i3]]).unwrap();

        let s_x =
            val1 - val3 + U::from(2).unwrap() * val4 - U::from(2).unwrap() * val6 + val7 - val8;
        let s_y =
            val1 + U::from(2).unwrap() * val2 + val3 - val7 - U::from(2).unwrap() * val8 - val9;

        let mag = (s_x * s_x + s_y * s_y).sqrt();

        *x = mag;
    });
    buffer
}

fn compute_energy<T>(img: &DynamicImage) -> Array2<T>
where
    T: Real + Sync + Send,
{
    let img = img.clone().into_luma8();
    let img: Array2<u8> = Array2::from_shape_vec(
        (img.height() as usize, img.width() as usize),
        img.as_raw().to_vec(),
    )
    .unwrap();
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

fn find_min_energy_path<T>(energy: &Array2<T>) -> Vec<usize>
where
    T: PartialOrd + Copy,
{
    let n_rows = energy.nrows();
    let w = energy.ncols();

    let mut idx = vec![0; n_rows];

    // TODO this -1 can panic
    idx[0] = energy
        .index_axis(Axis(0), n_rows - 1)
        .indexed_iter()
        .min_by(|(_, &x), (_, &y)| x.partial_cmp(&y).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    for (i, row) in energy.axis_iter(Axis(0)).rev().enumerate().skip(1) {
        let last_i = idx[i - 1];
        let min_i = last_i.saturating_sub(1);
        let max_i = w.min(last_i + 2) as usize;

        let temp = row
            .slice(s![min_i..max_i])
            .indexed_iter()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        idx[i] = temp + min_i;
    }

    idx.reverse();
    idx
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
