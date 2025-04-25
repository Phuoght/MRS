# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import csv

class PhimMoiPipeline:
    def open_spider(self, spider):
        if spider.name == "phimmoi":

            self.file = open("Movie_Phimmoi.csv", "a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file,
                                        fieldnames=['url_film', 'title', 'describe','rating', 
                                                    'rate_count', 'status', 'release_year', 
                                                    'country', 'genre', 'director', 'actor', 
                                                    'img_film'])
            
            if self.file.tell() == 0:
                self.writer.writeheader()

    def process_item(self, item, spider):
        if spider.name == "phimmoi":
            self.writer.writerow(item)
        return item

    def close_spider(self, spider):
        if spider.name == "phimmoi":
            self.file.close()