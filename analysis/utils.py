from PIL import Image

def create_image_grid(image_paths, grid_size, save_path="grid.png"):
    # Assuming all images are the same size, open the first image to get the size
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size

    # Calculate the size of the grid
    grid_width = img_width * grid_size[1]
    grid_height = img_height * grid_size[0]

    # Create a new image with a white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Paste images into the grid
    for index, image_path in enumerate(image_paths):
        row = index // grid_size[1]
        col = index % grid_size[1]
        with Image.open(image_path) as img:
            grid_image.paste(img, (col * img_width, row * img_height))

    # Save the grid image
    grid_image.save(save_path)
    
