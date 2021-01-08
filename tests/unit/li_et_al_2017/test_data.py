import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche.data import DownloadableImageCollection


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)
