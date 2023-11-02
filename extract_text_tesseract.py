def return_text(filename):
    import pytesseract
    from PIL import Image
# Open the image file
    image = Image.open(filename)

    # Perform OCR using PyTesseract
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    return(str(text))