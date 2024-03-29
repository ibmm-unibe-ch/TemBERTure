from PIL import Image, ImageDraw, ImageFont

def combine_images(image_path1, image_path2, output_path):
    width_ratio = 0.11  # Porzione dell'immagine che la larghezza del testo do
    font_family = "/home/rodelc/Downloads/Arial.ttf"
    top_margin = 100
    

    image1 = Image.open(image_path1)
    image1_name = image_path1.split('_non-thermo')[0]
    image2 = Image.open(image_path2)
    image2_name = image_path2.split('_thermo')[0]

    width1, height1 = image1.size
    width2, height2 = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    combined_image = Image.new('RGB', (result_width, result_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))

    combined_image.save(output_path)
    
    def get_text_size(text, image, font):
        image = Image.new('RGB', (image.width, image.height))
        return editable_image.textsize(text, font)
    
    def find_font_size(text, font, image, target_width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = get_text_size(text, image, tested_font)
        estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
        return round(estimated_font_size)
    
    image = Image.open(output_path)
    editable_image = ImageDraw.Draw(image)
    font_size = find_font_size(image2_name, font_family, image, width_ratio)
    font = ImageFont.truetype(font_family, font_size)
    
    width, height = image.size
    x1 = width / 4 - get_text_size(image2_name, image, font)[0] / 2
    x2 = 3 * width / 4 - get_text_size(image2_name, image, font)[0] / 2
    
    ## very tall protein, to avoid overlap between pdb id and structure
    if image_path2 =='1TML_A_thermo.png':
        editable_image.text((x1-650, top_margin), image1_name, font=font, fill=(0,0,0,255))
        editable_image.text((x2-650, top_margin), image2_name, font=font, fill=(0,0,0,255))
    elif image_path1=='1B0A_A_non-thermo.png':
        editable_image.text((x1+350, top_margin), image1_name, font=font, fill=(0,0,0,255))
        editable_image.text((x2, top_margin), image2_name, font=font, fill=(0,0,0,255))
    else:
        editable_image.text((x1, top_margin), image1_name, font=font, fill=(0,0,0,255))
        editable_image.text((x2, top_margin), image2_name, font=font, fill=(0,0,0,255))
    

    image.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python combine_images.py image1.png image2.png output.png")
        sys.exit(1)

    combine_images(sys.argv[1], sys.argv[2], sys.argv[3])

