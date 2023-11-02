def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # return texts
    # print("Texts:")
    return(str(texts))

    # for text in texts:
        # text = ""
        # text+=(f'\n"{text.description}"')

        # vertices = [
        #     f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        # ]

        # text+=("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )