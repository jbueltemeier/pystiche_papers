from urllib.parse import urljoin

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
)


__all__ = [
    "images",
]


def images() -> DownloadableImageCollection:
    root = (
        "https://raw.githubusercontent.com/Yijunmaverick/UniversalStyleTransfer/master/input/"
    )

    content_root = urljoin(root, "content/")
    content_images = {
        "women1": DownloadableImage(
            urljoin(content_root, "004.jpg"), md5="7a7720dccb3c1139a8d26f1bed795846"
        ),
        "flower": DownloadableImage(
            urljoin(content_root, "006.jpg"),
            md5="9b3ac3a8bae76cc9cce3af86dd2df3b9",
        ),
        "vincent": DownloadableImage(
            urljoin(content_root, "023.jpg"),
            md5="ff217acb6db32785b8651a0e316aeab3",
        ),
        "women2": DownloadableImage(
            urljoin(content_root, "028.jpg"),
            md5="981b6ecfb0bed1f6348989604203eb9e",
        ),
        "women3": DownloadableImage(
            urljoin(content_root, "04.jpg"),
            md5="a484e56dd4f252c4d675c465526f3996",
        ),
        "bridge": DownloadableImage(
            urljoin(content_root, "05.jpg"),
            md5="be29eaba58e6c5c691110ac52d553d05",
        ),
        "tubingen": DownloadableImage(
            urljoin(content_root, "tubingen.jpg"),
            md5="dc9ad203263f34352e18bc29b03e1066",
        ),
    }

    style_root = urljoin(root, "style/")
    style_images = {
        "abstract": DownloadableImage(
            urljoin(style_root, "002937.jpg"), md5="cd7ef1ddb4fbbd4f1020978548f067e7"
        ),
        "water": DownloadableImage(
            urljoin(style_root, "018.jpg"), md5="a1bef602781153a4220f78c4b2e94196"
        ),
        "tiger": DownloadableImage(
            urljoin(style_root, "088.jpg"), md5="7dfe7c55a91cbaa140618f5029a49573"
        ),
        "iron_art": DownloadableImage(
            urljoin(style_root, "09.jpg"), md5="365c73d8d3e30e2f94c023ccf3475d67"
        ),
        "women_painting": DownloadableImage(
            urljoin(style_root, "876.jpg"), md5="abc175dc6b11199944579248037ffc2e"
        ),
        "antimonocromatismo": DownloadableImage(
            urljoin(style_root, "antimonocromatismo.jpg"),
            md5="2845a7825f759cb2a00c49be69fcdd18"
        ),
        "brick": DownloadableImage(
            urljoin(style_root, "brick.jpg"), md5="daed7b5ed5a6151f87cafbf793ddfa9d"
        ),
        "brick1": DownloadableImage(
            urljoin(style_root, "brick1.jpg"), md5="ea91a4e8f06e9fff4f48385068e50b81"
        ),
        "seated_nude": DownloadableImage(
            urljoin(style_root, "seated-nude.jpg"),
            md5="4d8b26218eac0c80c578cca5c5204184"
        ),
        "women_hat": DownloadableImage(
            urljoin(style_root, "woman-with-hat-matisse.jpg"),
            md5="8897a19069a9305f60d71d3564828c66"
        ),
        "women_dress": DownloadableImage(
            urljoin(style_root, "woman_in_peasant_dress_cropped.jpg"),
            md5="53668dfa3b8a110bfd2c214c91c8cdf1"
        ),

    }
    return DownloadableImageCollection({**content_images, **style_images},)
