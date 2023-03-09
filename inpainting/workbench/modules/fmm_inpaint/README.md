# Rust Implementation of Image Inpainting Algorithms

This project is a Rust implementation of the inpainting algorithms based in the papers "An Image Inpainting Technique Based on the Fast Marching Method" by Alexandru Telea and "Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting" by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro.

## Cool... But what is Image Inpainting?

Image inpainting is the process of filling in missing or damaged parts of an image. This technique is useful for repairing images damaged by scratches, tears, or other forms of degradation. Inpainting algorithms can also be used to remove unwanted objects from an image.

## Implementation

This project provides a pure Rust implementation of the inpainting algorithms described in the mentioned papers. The implementation is based on the Fast Marching Method and uses techniques such as diffusion and texture synthesis to fill in missing areas of an image.

It is important to note that this project is still in development and is not yet ready for production use. The inpainting algorithms are not yet fully implemented and the user interface is still very basic.

This project aims to provide a fast and easy to use implementation of the inpainting algorithms. The goal is to provide a library that can be used to easily implement inpainting algorithms in other projects.

## Installation

To use this project, you will need to have Rust installed on your system. Once Rust is installed, you can clone the repository and run the program using the following commands:

```sh
git clone https://github.com/becelli/academy.git
cd academy/inpainting/fmm-inpaint
cargo run --release
```

## License

Currently this project is not licensed under any license. If you would like to use this project, please contact me at `gustavobecelli@gmail.com`

## Acknowledgments

This project is based on the research of Alexandru Telea and Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro. The original papers can be found here:

- [An Image Inpainting Technique Based on the Fast Marching Method](https://doi.org/10.1080/10867651.2004.10487596)

- [Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting](https://doi.org/10.1109/CVPR.2001.990497)
