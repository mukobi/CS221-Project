# RAISE - Raw Image Dataset

Download raw images from http://loki.disi.unitn.it/RAISE/download.html

## Preprocessing steps

### Downloading

Use `download_raise.py` to download the images from the RAISE dataset given a raise.csv file of urls next to it (download csv from the RAISE site above).

Note: for RAISE 2K (2000/8156 images from the dataset which is what we used), this is a 42.5 GB download and took 7.5 hours over the Stanford University residential ethernet network.

### Converting

To convert the raw images to jpg compressed files (smaller and better compatibility with Python's Pillow library), we used [ImageMagick](https://imagemagick.org/). Similar programs should work as well, but we recommend ImageMagick to replicate our same results and because it is simple to use.

In order to resize all the .TIF raw images to a resolution where the minimum dimension is 650 and convert them to 80% quality jpg images, run

```bash
magick mogrify -resize "650x650^" -format jpg -quality 80 *.TIF
```

When this is finished, you will notice ImageMagick also creates very-low resolution thumbnails suffixed "-1.jpg". We don't need them, so remove them with

```bash
rm -f *-1.jpg
```

We can also remove the original .TIF files. Remove them with

```bash
rm -f *.TIF
```

Now the photographic images from the RAISE Raw Image Dataset are processed and ready to learn on!