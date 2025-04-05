import time
from random import randint
import requests
from scrapy import signals


class SphinxValueVisionSpiderMiddleware:
    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        return None

    def process_spider_output(self, response, result, spider):
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        pass

    def process_start_requests(self, start_requests, spider):
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class SphinxValueVisionDownloaderMiddleware:
    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        return None

    def process_response(self, request, response, spider):
        return response

    def process_exception(self, request, exception, spider):
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class ScrapeOpsHeadersMiddleware:
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def __init__(self, settings):
        self.scrape_ops_api_key = settings.get('SCRAPE_OPS_API_KEY')
        self.scrape_ops_endpoint = settings.get('SCRAPE_OPS_FAKE_BROWSER_HEADER_ENDPOINT',
                                                 'https://headers.scrapeops.io/v1/browser-headers')
        self.scrape_ops_fake_browser_headers_active = settings.get('SCRAPE_OPS_FAKE_BROWSER_HEADER_ENABLED', True)
        self.scrape_ops_num_results = settings.get('SCRAPE_OPS_NUM_RESULTS')
        self.headers_list = []
        self._scrapeops_fake_browser_headers_enabled()
        self._get_headers_list_with_backoff()

    def _scrapeops_fake_browser_headers_enabled(self):
        if (not self.scrape_ops_api_key or not self.scrape_ops_fake_browser_headers_active):
            self.scrape_ops_fake_browser_headers_active = False
        else:
            self.scrape_ops_fake_browser_headers_active = True

    def _get_headers_list_with_backoff(self):
        """
        Retrieve headers from ScrapeOps API with basic exponential backoff to avoid 429 errors.
        """
        payload = {'api_key': self.scrape_ops_api_key}
        if self.scrape_ops_num_results is not None:
            payload['num_results'] = self.scrape_ops_num_results

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.scrape_ops_endpoint, params=payload)
                if response.status_code == 200:
                    json_response = response.json()
                    self.headers_list = json_response.get('result', [])
                    if self.headers_list:
                        break  # Successfully retrieved headers
                elif response.status_code == 429:
                    wait_time = 2 ** attempt  # Exponential backoff (2, 4, 8 seconds)
                    print(f"Received 429 rate limit. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch headers. Status: {response.status_code}, Response: {response.text}")
                    break
            except requests.RequestException as e:
                print("Request error while fetching headers:", str(e))
                break

    def _get_random_browser_header(self):
        if not self.headers_list:
            # In case headers_list is empty, return an empty dictionary
            return {}
        random_index = randint(0, len(self.headers_list) - 1)
        return self.headers_list[random_index]

    def process_request(self, request, spider):
        if not self.scrape_ops_fake_browser_headers_active:
            return
        random_browser_header = self._get_random_browser_header()

        # Define the header keys we want to update
        header_keys = [
            'accept-language',
            'sec-fetch-user',
            'sec-fetch-mod',
            'sec-fetch-site',
            'sec-ch-ua-platform',
            'sec-ch-ua-mobile',
            'sec-ch-ua',
            'accept',
            'user-agent',
            'upgrade-insecure-requests',
        ]

        for key in header_keys:
            value = random_browser_header.get(key)
            if value:
                request.headers[key] = value


class InfiniteExponentialBackoffMiddleware:
    # Configure the initial and maximum delay (in seconds)
    initial_delay = 10
    max_delay = 600

    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        # Optionally, you can pull these values from settings:
        middleware.initial_delay = crawler.settings.getint('RETRY_INITIAL_DELAY', 10)
        middleware.max_delay = crawler.settings.getint('RETRY_MAX_DELAY', 600)
        return middleware

    def spider_opened(self, spider):
        spider.logger.info("InfiniteExponentialBackoffMiddleware enabled.")

    def process_response(self, request, response, spider):
        # Check if the response status code is 429 (Too Many Requests)
        if response.status == 429:
            # Get the current retry count; default is 0 if not set
            retry_count = request.meta.get('retry_count', 0)
            # Calculate exponential backoff delay
            delay = min(self.initial_delay * (2 ** retry_count), self.max_delay)
            spider.logger.info(
                f"Received 429 error. Retry count: {retry_count}. Sleeping for {delay} seconds before retrying..."
            )
            # Wait for the calculated delay (blocking call)
            time.sleep(delay)
            # Increment retry count and copy the request for a retry
            new_request = request.copy()
            new_request.meta['retry_count'] = retry_count + 1
            # Return the new request to retry without stopping the spider
            return new_request

        # For all other responses, simply return the response
        return response

    def process_exception(self, request, exception, spider):
        # Optionally, add custom exception handling if needed
        return None