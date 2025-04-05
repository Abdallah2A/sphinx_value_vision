import scrapy


class AqarmapSpider(scrapy.Spider):
    name = 'aqarmap'
    start_urls = ['https://aqarmap.com.eg/en/for-sale/apartment/?page=1&deliveryYear=0']

    def parse(self, response, **kwargs):
        # Get current page from meta, default to 1 for initial request
        current_page = response.meta.get('current_page', 1)

        # Extract apartment links
        apartment_urls = self.extract_apartment_links(response)

        if not apartment_urls:
            self.logger.info(f"No apartments found on page {current_page}, stopping crawl.")
            return

        # Process apartment pages
        for url in apartment_urls:
            url = f"https://aqarmap.com.eg{url}"
            yield scrapy.Request(url, callback=self.parse_apartment)

        # Generate next page request
        next_page = current_page + 1
        next_url = f"https://aqarmap.com.eg/en/for-sale/apartment/?page={next_page}&deliveryYear=0"
        yield scrapy.Request(
            next_url,
            callback=self.parse,
            meta={'current_page': next_page}
        )

    @staticmethod
    def extract_apartment_links(response):
        links = response.css('div.listing-card.h-auto a::attr(href)').getall()
        return list(set(links))

    def parse_apartment(self, response):
        apartment_data = {
            'url': response.url,
            'price': self.extract_apartment_price(response),
            'location': self.extract_apartment_location(response),
            **self.extract_property_details(response),
            **self.extract_listing_vars(response)
        }

        if all(apartment_data.values()):
            yield apartment_data
        else:
            self.logger.debug(f"Skipping incomplete item: {response.url}")

    @staticmethod
    def extract_apartment_price(response):
        # First method
        price = response.css('div.flex-1 span::text').get()
        if price:
            return ' '.join(price.split())

        # Second method
        price_parts = response.css(
            'div.listing-details-page__title-section__price span::text'
        ).getall()

        if price_parts:
            cleaned = [p.strip() for p in price_parts if p.strip()]
            return ' '.join(cleaned)

        return None

    @staticmethod
    def extract_apartment_location(response):
        location = response.css("li[aria-current='page'] span::text").get()
        if location:
            location = location.strip().split('sale in ')[-1]
            return location

        location = response.css('p.truncated-text.text-body_2::text').get()
        return location.strip() if location else None

    @staticmethod
    def extract_property_details(response):
        details = response.css('p.truncated-text.text-body_1::text').getall()
        details = [d.strip() for d in details if d.strip()]

        return {
            'area': details[0] if len(details) > 0 else None,
            'rooms': details[1] if len(details) > 1 else None,
            'bathrooms': details[2] if len(details) > 2 else None,
            'style': details[3] if len(details) > 3 else None
        }

    @staticmethod
    def extract_listing_vars(response):
        vars_data = {
            'floor': None,
            'year_built': None,
            'seller_type': None,
            'view': None,
            'payment_method': None
        }

        groups = response.xpath('//div[contains(@class, "group") and contains(@class, "flex")]')

        for group in groups:
            key = group.xpath('./*[1]//text()').get('').strip()
            value = group.xpath('./*[2]//text()').get('').strip()

            if key == 'Floor':
                vars_data['floor'] = value
            elif key == 'Year Built':
                vars_data['year_built'] = value
            elif key == 'Seller Type':
                vars_data['seller_type'] = value
            elif key == 'View':
                vars_data['view'] = value
            elif key == 'Payment Method':
                vars_data['payment_method'] = value

        return vars_data
