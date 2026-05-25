import asyncio
from playwright.async_api import async_playwright
import os

async def move_and_click(page, locator, pause_before=0.4):
    await locator.scroll_into_view_if_needed()
    box = await locator.bounding_box()
    if box:
        x = box['x'] + box['width'] / 2
        y = box['y'] + box['height'] / 2
        await page.mouse.move(x, y, steps=25)
        await asyncio.sleep(pause_before)
        await page.mouse.click(x, y)
    else:
        await locator.click()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        # Go to Streamlit
        print("1. Navigate to Streamlit...")
        await page.goto("http://127.0.0.1:8501")
        await page.wait_for_timeout(5000)
        
        # Simulate Section 4: reload
        print("2. Reload page...")
        await page.reload()
        await page.wait_for_timeout(4000)
        await page.screenshot(path="scratch/sec4_step1_reloaded.png")
        
        # Re-locate tabs
        tabs = page.locator("button[role='tab']")
        print(f"Tabs count: {await tabs.count()}")
        
        # Click Tab 3
        print("3. Clicking Tab 3...")
        await move_and_click(page, tabs.nth(2), pause_before=0.4)
        await page.wait_for_timeout(3000)
        await page.screenshot(path="scratch/sec4_step2_tab3_clicked.png")
        
        # Locate input
        print("4. Locating input...")
        inspector_input = page.get_by_label("Enter Job UUID Index to Query:")
        print("Is input visible:", await inspector_input.is_visible())
        
        # Fill input
        job_id_text = "65afa217-6106-47b4-86f9-c736d78c0438"
        print("5. Filling input...")
        await move_and_click(page, inspector_input, pause_before=0.4)
        await inspector_input.fill(job_id_text)
        await page.screenshot(path="scratch/sec4_step3_filled.png")
        
        # Press Enter
        print("6. Pressing Enter...")
        await inspector_input.press("Enter")
        await page.wait_for_timeout(8000)
        await page.screenshot(path="scratch/sec4_step4_after_enter.png")
        
        # Check text
        print("Page text content:")
        body_text = await page.locator("body").inner_text()
        print(body_text[:1000]) # Print first 1000 chars
        
        # Check canvas/svg
        chart = page.locator("[data-testid='stArrowVegaLiteChart']")
        print("Chart count:", await chart.count())
        if await chart.count() > 0:
            print("Chart visible:", await chart.first.is_visible())
            svg = chart.first.locator("svg")
            print("SVG count inside chart:", await svg.count())
            
        await browser.close()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
