from pystiche import data

__all__ = ["images"]


def images() -> data.DownloadableImageCollection:
    images_ = {
        "neckarfront": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
            title="Tübingen Neckarfront",
            author="Andreas Praefcke",
            license=data.CreativeCommonsLicense(("by",), version="3.0"),
            md5="dc9ad203263f34352e18bc29b03e1066",
            file="tuebingen_neckarfront__andreas_praefcke.jpg",
        ),
        "shipwreck": data.DownloadableImage(
            "https://blog-imgs-51.fc2.com/b/e/l/bell1976brain/800px-Shipwreck_turner.jpg",
            title="Shipwreck of the Minotaur",
            author="J. M. W. Turner",
            date="ca. 1810",
            license=data.ExpiredCopyrightLicense(1851),
            md5="4fb76d6f6fc1678cb74e858324d4d0cb",
            file="shipwreck_of_the_minotaur__turner.jpg",
        ),
        "starry_night": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            title="Starry Night",
            author="Vincent van Gogh",
            date="ca. 1889",
            license=data.ExpiredCopyrightLicense(1890),
            md5="372e5bc438e3e8d0eb52cc6f7ef44760",
        ),
        "the_scream": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
            title="The Scream",
            author="Edvard Munch",
            date="ca. 1893",
            license=data.ExpiredCopyrightLicense(1944),
            md5="46ef64eea5a7b2d13dbadd420b531249",
        ),
        "femme_nue_assise": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg",
            title="Figure dans un Fauteuil",
            author="Pablo Ruiz Picasso",
            date="ca. 1909",
            license=data.ExpiredCopyrightLicense(1973),
            md5="ba14b947b225d9e5c59520a814376944",
        ),
        "composition_vii": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
            title="Composition VII",
            author="Wassily Kandinsky",
            date="1913",
            license=data.ExpiredCopyrightLicense(1944),
            md5="bfcbc420684bf27d2d8581fa8cc9522f",
        ),
    }
    return data.DownloadableImageCollection(images_)
