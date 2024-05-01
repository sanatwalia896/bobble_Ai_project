**README.md**

# Face Morphing using StyleGAN

This project implements a face morphing algorithm using StyleGAN, a generative adversarial network (GAN) architecture, to generate high-quality and realistic morphed images. The algorithm allows users to blend features from two input images seamlessly, creating smooth transitions between facial attributes.

## Dependencies

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- dlib
- Ninja

## Usage

1. Upload the source and target images.
2. Run the provided code to preprocess the images and project them into the StyleGAN latent space.
3. Adjust the interpolation factor to control the degree of blending between the source and target images.
4. Generate the morphed image using StyleGAN.
5. Visualize and save the morphed image.

## Preprocessed Images

### Source Image
![Source Image](images/source_preprocessed.png)

### Target Image
![Target Image]([images/target_preprocessed.png](https://github.com/sanatwalia896/bobble_Ai_project/blob/main/image5.webp))

## StyleGAN Latent Space

### Source Image
![Source Latent](images/source_latent.png)

### Target Image
![Target Latent](images/target_latent.png)

## Morphed Image

![Morphed Image](images/morphed_image.png)

## Credits

This project was inspired by and adapted from the code provided by Jeff Heaton in his [StyleGAN2-ADA-PyTorch repository](https://github.com/NVlabs/stylegan2-ada-pytorch).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
