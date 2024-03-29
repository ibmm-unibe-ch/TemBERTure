from PIL import Image

def create_image_grid(images, rows, cols, output_path):
    # Ensure there are enough images to fill the grid
    #assert len(images) >= rows * cols, "Not enough images to fill the grid"

    # Open the images and calculate the maximum width and height of a cell
    imgs = [Image.open(image) for image in images]
    max_width = max(img.size[0] for img in imgs)
    max_height = max(img.size[1] for img in imgs)

    # Create a new image with the appropriate height and width
    background_color = (255, 255, 255)  # RGB per il bianco
    grid_img = Image.new('RGB', (cols * max_width, rows * max_height), background_color)

    # Paste the images into the grid
    for i, img in enumerate(imgs):
        if i >= rows * cols:  # Stop if the grid is full
            break
        grid_x = i % cols * max_width
        grid_y = i // cols * max_height
        grid_img.paste(img, (grid_x, grid_y))

    # Save the grid image
    print('Saving..')
    grid_img.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python create_image_grid.py output.png row col image1.png image2.png ...")
        sys.exit(1)

    output_file = sys.argv[1]
    row = int(sys.argv[2])
    col = int(sys.argv[3])
    image_files = sys.argv[4:]
    
    create_image_grid(image_files, row, col, output_file)


