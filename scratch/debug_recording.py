import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        print("Launching headless Chromium...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        # 1. Navigate to Streamlit App
        print("Navigating to http://127.0.0.1:8501...")
        await page.goto("http://127.0.0.1:8501")
        await page.wait_for_timeout(5000)
        
        # Take initial screenshot
        await page.screenshot(path="scratch/debug_step1.png")
        print("Saved debug_step1.png")
        
        # 2. Go to Tab 3
        tabs = page.locator("button[role='tab']")
        print("Clicking Tab 3...")
        await tabs.nth(2).click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path="scratch/debug_step2.png")
        print("Saved debug_step2.png")
        
        # 3. Type job ID
        job_id_text = "65afa217-6106-47b4-86f9-c736d78c0438"
        inspector_input = page.get_by_label("Enter Job UUID Index to Query:")
        print("Typing job ID...")
        await inspector_input.fill(job_id_text)
        await page.wait_for_timeout(1000)
        await inspector_input.press("Enter")
        await page.wait_for_timeout(5000)
        await page.screenshot(path="scratch/debug_step3.png")
        print("Saved debug_step3.png")
        
        # 4. Reload page (simulate Section 4 reload)
        print("Reloading page...")
        await page.reload()
        await page.wait_for_timeout(5000)
        await page.screenshot(path="scratch/debug_step4_after_reload.png")
        print("Saved debug_step4_after_reload.png")
        
        # 5. Click Tab 3 again
        tabs = page.locator("button[role='tab']")
        print("Clicking Tab 3 again...")
        await tabs.nth(2).click()
        await page.wait_for_timeout(3000)
        await page.screenshot(path="scratch/debug_step5_tab3_clicked.png")
        print("Saved debug_step5_tab3_clicked.png")
        
        # 6. Type job ID again
        inspector_input = page.get_by_label("Enter Job UUID Index to Query:")
        print("Typing job ID again...")
        await inspector_input.fill(job_id_text)
        await page.wait_for_timeout(1000)
        await inspector_input.press("Enter")
        await page.wait_for_timeout(10000) # Give it 10s
        await page.screenshot(path="scratch/debug_step6_after_enter.png")
        print("Saved debug_step6_after_enter.png")
        
        # Check if canvas exists
        canvases = page.locator("canvas")
        count = await canvases.count()
        print(f"Number of canvas elements found: {count}")
        
        await browser.close()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
