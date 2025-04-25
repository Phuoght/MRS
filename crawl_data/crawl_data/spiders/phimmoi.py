import scrapy
from crawl_data.items import CrawlDataItem

class PhimmoiSpider(scrapy.Spider):
    name = "phimmoi"
    allowed_domains = ["phimmoichill.blog"]
    start_urls = ["https://phimmoichill.blog"]

    def start_requests(self):
        genres = ['phim-hanh-dong', 'phim-tinh-cam', 'phim-hai-huoc', 'phim-co-trang', 
                  'phim-tam-ly', 'phim-hinh-su', 'phim-chien-tranh', 'phim-the-thao', 
                  'phim-vo-thuat', 'phim-hoat-hinh', 'phim-vien-tuong', 'phim-phieu-luu', 
                  'phim-khoa-hoc', 'phim-ma-kinh-di', 'phim-am-nhac', 'phim-than-thoai']
        urls = []
        for genre in genres:
            domain = f'https://phimmoichill.blog/genre/{genre}'
            urls.append(domain)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_link)
            for i in range(2, 141):
                link = f'{url}/page-{i}'
                yield scrapy.Request(url=link, callback=self.parse_link)

    def parse_link(self, response):
        #binlist > ul > li:nth-child(1) > a  
        #binlist > ul > li:nth-child(2) > a
        if response.css('#binlist > ul::text').get() == ' Không có phim nào ! Bạn có thể tìm bằng tên tiếng anh hoặc từ khóa không dấu.':
            return
        else:
            for i in range(1, 26):
                selector = f'#binlist > ul > li:nth-child({i}) > a::attr(href)'
                #binlist > ul > li:nth-child(1) > a
                link = response.css(selector).extract_first()

                yield scrapy.Request(url=link, callback=self.parse)
            
    def parse(self, response):
        item = CrawlDataItem()
        item['url_film'] = response.url
        item['img_film'] = response.css('#detail-page > div.film-info > div.image > img::attr(src)').get()
        item['title']  = response.css('#detail-page > div.film-info > div.image > div > h1::text').get()
        item['describe'] = response.css('#film-content::text').getall()
        item['rating'] = response.css('#star::attr(data-score)').get()
        item['rate_count'] = response.css('#rate_count::text').get()
        
        if response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(1) > label::text').get() == 'Đang phát: ':
            item['status'] = 1
        else:
            item['status'] = 0
        
        item['release_year'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(2) > a::text').get()
        item['country'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(3) > a::text').get()
        item['genre'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(4) > a:nth-child(2)::text').get()
        item['director'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(5) > span > a > span::text').get()
        
        if response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(8) > a::text').getall() == []:
            if response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(9) > a::text').getall() == []:
                item['actor'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(10) > a::text').getall()
            else:
                item['actor'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(9) > a::text').getall()
        else:
            item['actor'] = response.css('#detail-page > div.film-info > div.text > ul > li:nth-child(8) > a::text').getall()
        yield item
