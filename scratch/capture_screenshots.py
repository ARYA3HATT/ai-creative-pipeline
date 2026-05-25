import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        # Launch headless browser
        print("Launching headless Chromium shell...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1300})
        
        # 1. Navigate to Streamlit App
        print("Navigating to Streamlit at http://localhost:8501...")
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000) # Let assets load
        
        # 2. Navigate to Tab 3: System Log & Metadata Inspector
        print("Navigating to Tab 3: System Log & Metadata Inspector...")
        tabs = page.locator("button[role='tab']")
        await tabs.nth(2).click() # Click Tab 3
        await page.wait_for_timeout(3000)
        
        # Enter Job ID for Test 2 (65afa217-6106-47b4-86f9-c736d78c0438)
        job_id_test2 = "65afa217-6106-47b4-86f9-c736d78c0438"
        print(f"Entering Job ID {job_id_test2} into Inspector...")
        input_field = page.get_by_label("Enter Job UUID Index to Query:")
        await input_field.wait_for(state="attached", timeout=10000)
        await input_field.fill(job_id_test2)
        await input_field.press("Enter")
        await page.wait_for_timeout(6000) # Let the chart and details render
        
        # Capture Tab 3 (VLM score progression chart and critiques)
        chart_screenshot_path = "/Users/priyanshu/.gemini/antigravity-ide/brain/149821d6-e7fe-4bb4-bdad-cc6e5cccba07/vlm_score_progression_chart_1779516178433.png"
        print(f"Saving VLM chart screenshot to {chart_screenshot_path}...")
        await page.screenshot(path=chart_screenshot_path, full_page=False)
        
        # 3. Navigate to Tab 1: Single URL Ingestion to load Pollinations.ai FLUX Gallery
        print("Navigating to Tab 1...")
        await tabs.nth(0).click()
        await page.wait_for_timeout(3000)
        
        # Input the Completed Job ID for Test 1 (a8aa6dec-ca99-4e5f-ad28-e34483043bfe)
        job_id_test1 = "a8aa6dec-ca99-4e5f-ad28-e34483043bfe"
        print(f"Loading completed Pollinations gallery for Job ID: {job_id_test1}...")
        load_input = page.get_by_label("Enter Completed Job ID to load gallery")
        await load_input.wait_for(state="attached", timeout=15000)
        await load_input.fill(job_id_test1)
        
        # Click "Load Completed Gallery"
        print("Clicking Load Completed Gallery...")
        await page.locator("button:has-text('Load Completed Gallery')").click()
        await page.wait_for_timeout(8000) # Let the gallery images extract and load
        
        # Capture the Streamlit Campaign Gallery showing real FLUX generated images
        gallery_screenshot_path = "/Users/priyanshu/.gemini/antigravity-ide/brain/149821d6-e7fe-4bb4-bdad-cc6e5cccba07/streamlit_gallery_screenshot_1779516158720.png"
        print(f"Saving Streamlit gallery screenshot to {gallery_screenshot_path}...")
        await page.screenshot(path=gallery_screenshot_path, full_page=False)
            
        # 4. Capture Tab 2: Bulk Ingestion
        print("Navigating to Tab 2: Bulk CSV Ingestion...")
        await tabs.nth(1).click()
        await page.wait_for_timeout(3000)
        
        # Take screenshot of the bulk dashboard
        bulk_screenshot_path = "/Users/priyanshu/.gemini/antigravity-ide/brain/149821d6-e7fe-4bb4-bdad-cc6e5cccba07/bulk_tracker_screenshot_1779516196154.png"
        print(f"Saving Bulk Tracker screenshot to {bulk_screenshot_path}...")
        await page.screenshot(path=bulk_screenshot_path)

        await browser.close()
        print("Screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(main())
