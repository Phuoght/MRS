# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlDataItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url_film = scrapy.Field()
    title = scrapy.Field()
    describe = scrapy.Field()
    rating = scrapy.Field()
    rate_count = scrapy.Field()
    status = scrapy.Field()
    release_year = scrapy.Field()
    country = scrapy.Field()
    genre = scrapy.Field()
    director = scrapy.Field()
    actor = scrapy.Field()
    img_film = scrapy.Field()

