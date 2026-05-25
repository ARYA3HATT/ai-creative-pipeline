class AsyncWebCrawler:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    async def arun(self, *args, **kwargs):
        class MockResult:
            success = False
            error_message = "Local Crawl4AI mock active - triggering BeautifulSoup fallback."
        return MockResult()
