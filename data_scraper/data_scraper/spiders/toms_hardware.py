import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from data_scraper.items import Scraper
import re

class TomsHardwareSpider(CrawlSpider):
    name = "tomshardware"
    allowed_domains = ["forums.tomshardware.com"]
    start_urls = []

    for pgnum in range(10, 50):
        start_urls.append(f"https://forums.tomshardware.com/forums/storage.8/page-40{pgnum}?order=reply_count&direction=asc&thread_type=question")
    
    restricted_to = ['//div[@class="p-body-content"]']
    rules = [Rule(LinkExtractor(allow=r'threads/.*', restrict_xpaths=restricted_to), callback='parse_info', follow=True)]

    def parse_info(self, response):
        scraper = Scraper()

        scraper['title'] = response.xpath('//h1[@class="p-title-value"]/text()').get()
        scraper['url'] = response.url

        count = 0
        usercomment_list = []

        for message in response.xpath('//article[contains(@class, "message--post")]//div[@class="bbWrapper"]').getall():
            cleaned_message = re.sub(r'<blockquote.*?</blockquote>', '', message, flags=re.DOTALL)
            cleaned_message = re.sub(r'<.*?>', '', cleaned_message, flags=re.DOTALL)
            cleaned_message = re.sub(r'\s\s+', ' ', cleaned_message, flags=re.DOTALL)
            cleaned_message = re.sub(r'\[quotemsg=\d+,\d+,\d+\].*?\[/quotemsg\]', '', cleaned_message, flags=re.DOTALL)

            user = response.xpath('//h4[@class="message-name"]//text()')[count].get()
            
            usercomment_list.append(user + "----" + cleaned_message)
            scraper['usercomment'] = usercomment_list

            count += 1
        
        return scraper
