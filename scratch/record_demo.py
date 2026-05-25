import asyncio
import subprocess
import os
import shutil
from playwright.async_api import async_playwright

async def move_and_hover(page, locator, duration=1.0):
    await locator.scroll_into_view_if_needed()
    box = await locator.bounding_box()
    if box:
        x = box['x'] + box['width'] / 2
        y = box['y'] + box['height'] / 2
        await page.mouse.move(x, y, steps=25)
        await asyncio.sleep(duration)

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

async def scroll_smoothly(page, target_y):
    await page.evaluate(f"window.scrollTo({{top: {target_y}, behavior: 'smooth'}})")
    await page.wait_for_timeout(800)

async def record():
    # 1. Start Python HTTP Server to serve README.md
    print("Starting local HTTP server on port 8080...")
    http_server = subprocess.Popen(
        ["python3", "-m", "http.server", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 2. Launch Playwright Headless Browser
    async with async_playwright() as p:
        print("Launching headless Chromium...")
        browser = await p.chromium.launch(headless=True)
        
        # Prepare context with video recording
        raw_video_dir = "scratch/raw_videos"
        os.makedirs(raw_video_dir, exist_ok=True)
        
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=raw_video_dir,
            record_video_size={"width": 1920, "height": 1080}
        )
        
        page = await context.new_page()
        
        # ==========================================
        # SECTION 1: Single URL Ingestion (0:00 to 0:54 - 54s)
        # ==========================================
        print("Starting Section 1: Single URL Ingestion...")
        start_time = asyncio.get_event_loop().time()
        
        # A. Navigate to Streamlit App
        await page.goto("http://127.0.0.1:8501")
        await page.wait_for_timeout(4000)
        
        # B. Slowly scroll down to show the full dashboard layout
        await scroll_smoothly(page, 400)
        await page.wait_for_timeout(1000)
        await scroll_smoothly(page, 800)
        await page.wait_for_timeout(1000)
        
        # C. Scroll back up
        await scroll_smoothly(page, 0)
        await page.wait_for_timeout(1000)
        
        # D. Hover over input, click and type the URL slowly
        target_input = page.get_by_placeholder("e.g. https://www.amazon.com/dp/B08H75RTZ8")
        await move_and_click(page, target_input, pause_before=0.4)
        await page.wait_for_timeout(500)
        
        url_text = "https://www.amazon.in/dp/B0CHX1W1XY"
        await target_input.press_sequentially(url_text, delay=50)
        await page.wait_for_timeout(1000)
        
        # E. Hover over Launch Campaign button for 600ms before clicking
        launch_btn = page.get_by_role("button", name="Launch Campaign")
        await move_and_click(page, launch_btn, pause_before=0.6)
        await page.wait_for_timeout(4000)
        
        # F. Scroll down slowly to reveal the progress bar
        await scroll_smoothly(page, 400)
        await page.wait_for_timeout(2000)
        
        # G. Load pre-completed campaign to render gallery instantly
        load_input = page.get_by_placeholder("e.g. e37f5aa4-81e1-4513-aa1c-287c92dd63da")
        await move_and_click(page, load_input, pause_before=0.4)
        await load_input.fill("a8aa6dec-ca99-4e5f-ad28-e34483043bfe")
        await page.wait_for_timeout(1000)
        
        load_btn = page.get_by_role("button", name="Load Completed Gallery")
        await move_and_click(page, load_btn, pause_before=0.4)
        await page.wait_for_timeout(3000)
        
        # Maintain exact timing of 54 seconds for Section 1
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 54.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # ==========================================
        # SECTION 2: Completed Campaign Gallery (0:54 to 2:42 - 108s)
        # ==========================================
        print("Starting Section 2: Completed Campaign Gallery...")
        start_time = asyncio.get_event_loop().time()
        
        # Scroll down slowly to reveal the image grid
        await scroll_smoothly(page, 550)
        await page.wait_for_timeout(2000)
        
        # Find images to hover
        images = page.locator("[data-testid='stImage'] img")
        
        # Hover over image 1, 2, 3 for 1s
        if await images.count() >= 3:
            print("Hovering over images 1, 2, 3...")
            await move_and_hover(page, images.nth(0), duration=1.0)
            await page.wait_for_timeout(500)
            await move_and_hover(page, images.nth(1), duration=1.0)
            await page.wait_for_timeout(500)
            await move_and_hover(page, images.nth(2), duration=1.0)
            await page.wait_for_timeout(500)
            
        # Scroll down further to reveal image 4 and 5, hover
        await scroll_smoothly(page, 950)
        await page.wait_for_timeout(2000)
        
        if await images.count() >= 5:
            print("Hovering over images 4 and 5...")
            await move_and_hover(page, images.nth(3), duration=1.0)
            await page.wait_for_timeout(500)
            await move_and_hover(page, images.nth(4), duration=1.0)
            await page.wait_for_timeout(500)
            
        # Scroll back up slightly then down again to show all 5 together
        await scroll_smoothly(page, 500)
        await page.wait_for_timeout(2000)
        await scroll_smoothly(page, 950)
        await page.wait_for_timeout(2000)
        
        # Continue scrolling down to reveal video section
        await scroll_smoothly(page, 1550)
        await page.wait_for_timeout(2000)
        
        # Hover over video 1 and video 2
        videos = page.locator("video")
        if await videos.count() >= 2:
            print("Hovering over video players...")
            await move_and_hover(page, videos.nth(0), duration=2.0)
            await page.wait_for_timeout(1000)
            # Hover over video 2
            await move_and_hover(page, videos.nth(1), duration=2.0)
            await page.wait_for_timeout(1000)
            
        # Scroll down to show both videos simultaneously
        await scroll_smoothly(page, 1850)
        await page.wait_for_timeout(3000)
        
        # Scroll back up to show full gallery one more time before moving on
        await scroll_smoothly(page, 750)
        await page.wait_for_timeout(3000)
        
        # Maintain exact timing of 108 seconds for Section 2
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 108.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # ==========================================
        # SECTION 3: Audit Inspector (2:42 to 4:30 - 108s)
        # ==========================================
        print("Starting Section 3: Audit Inspector...")
        start_time = asyncio.get_event_loop().time()
        
        # Switch to Tab 3 (Inspector)
        tabs = page.locator("button[role='tab']")
        await move_and_click(page, tabs.nth(2), pause_before=0.4)
        await page.wait_for_timeout(3000)
        
        # Paste job ID slowly character by character
        inspector_input = page.get_by_label("Enter Job UUID Index to Query:")
        await move_and_click(page, inspector_input, pause_before=0.4)
        
        job_id_text = "65afa217-6106-47b4-86f9-c736d78c0438"
        await inspector_input.fill(job_id_text)
        await page.wait_for_timeout(1000)
        await inspector_input.press("Enter")
        await page.wait_for_timeout(8000) # Let blocks load
        
        # Scroll down slowly
        await scroll_smoothly(page, 300)
        await page.wait_for_timeout(1000)
        
        # Click to expand Product Research block
        expander1 = page.get_by_text("Expandable Section 1")
        await move_and_click(page, expander1, pause_before=0.4)
        await page.wait_for_timeout(1500) # Pause 1.5s to read
        
        # Scroll down, expand Creative Brief block
        await scroll_smoothly(page, 700)
        expander2 = page.get_by_text("Expandable Section 2")
        await move_and_click(page, expander2, pause_before=0.4)
        await page.wait_for_timeout(1500) # Pause 1.5s
        
        # Scroll down, expand Prompts block
        await scroll_smoothly(page, 1100)
        expander3 = page.get_by_text("Expandable Section 3")
        await move_and_click(page, expander3, pause_before=0.4)
        await page.wait_for_timeout(1500) # Pause 1.5s
        
        # Scroll down to metadata.json block
        await scroll_smoothly(page, 1500)
        await page.wait_for_timeout(2000)
        
        # Hover over qa_status: PASSED for 2s
        passed_text = page.get_by_text("PASSED").first
        await move_and_hover(page, passed_text, duration=2.0)
        await page.wait_for_timeout(500)
        
        # Hover over score 8.25 for 2s
        score_text = page.get_by_text("8.25").first
        await move_and_hover(page, score_text, duration=2.0)
        await page.wait_for_timeout(1000)
        
        # Slowly scroll back to top of inspector
        await scroll_smoothly(page, 800)
        await page.wait_for_timeout(1000)
        await scroll_smoothly(page, 0)
        await page.wait_for_timeout(2000)
        
        # Maintain exact timing of 108 seconds for Section 3
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 108.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # ==========================================
        # SECTION 4: Retry loop (4:30 to 5:51 - 81s)
        # ==========================================
        print("Starting Section 4: Retry Loop...")
        start_time = asyncio.get_event_loop().time()
        
        # Click back to Single URL tab
        await move_and_click(page, tabs.nth(0), pause_before=0.4)
        await page.wait_for_timeout(3000)
        
        # Clear input field, type retry URL slowly
        await move_and_click(page, target_input, pause_before=0.4)
        await target_input.fill("")
        await page.wait_for_timeout(1000)
        
        retry_url_text = "https://www.amazon.in/dp/B0CHX1W1XY?retry=true"
        await target_input.press_sequentially(retry_url_text, delay=50)
        await page.wait_for_timeout(1000)
        
        # Hit Launch Campaign
        await move_and_click(page, launch_btn, pause_before=0.6)
        await page.wait_for_timeout(3000)
        
        # Scroll down to show progress bar cycling
        await scroll_smoothly(page, 400)
        await page.wait_for_timeout(8000)
        
        # Stop background polling loop by refreshing the page
        print("Refreshing page to cleanly stop polling loop...")
        await page.reload()
        await page.wait_for_timeout(4000)
        
        # Re-locate elements after reload
        tabs = page.locator("button[role='tab']")
        
        # Navigate to Audit Inspector
        await move_and_click(page, tabs.nth(2), pause_before=0.4)
        await page.wait_for_timeout(3000)
        
        # Locate input handle after Tab 3 loads
        inspector_input = page.get_by_label("Enter Job UUID Index to Query:")
        
        await move_and_click(page, inspector_input, pause_before=0.4)
        await inspector_input.fill(job_id_text)
        await page.wait_for_timeout(1000)
        await inspector_input.press("Enter")
        await page.wait_for_timeout(8000)
        
        # Scroll down directly to the VLM Score Progression Chart
        # Since expanders 1, 2, 3 are collapsed, chart will be visible at y=450
        await scroll_smoothly(page, 450)
        await page.wait_for_timeout(2000)
        
        # Hover over each bar in the chart (Attempt 1: 6.25, Attempt 2: 6.25, Attempt 3: 8.25)
        # Using the vega chart SVG selector since Streamlit renders charts as SVG
        canvas = page.locator("svg.marks").first
        await canvas.wait_for(state="visible", timeout=25000)
        box = await canvas.bounding_box()
        if box:
            print("Hovering over chart bars...")
            # Bar 1 (6.25)
            await page.mouse.move(box['x'] + box['width']/6, box['y'] + box['height']/2, steps=25)
            await asyncio.sleep(1.5)
            # Bar 2 (6.25)
            await page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2, steps=25)
            await asyncio.sleep(1.5)
            # Bar 3 (8.25)
            await page.mouse.move(box['x'] + 5*box['width']/6, box['y'] + box['height']/2, steps=25)
            await asyncio.sleep(1.5)
            
        # Scroll down to show prompt mutation details between attempts
        await scroll_smoothly(page, 850)
        await page.wait_for_timeout(4000)
        
        # Maintain exact timing of 81 seconds for Section 4
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 81.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # ==========================================
        # SECTION 5: Bulk CSV Ingestion (5:51 to 7:39 - 108s)
        # ==========================================
        print("Starting Section 5: Bulk Ingestion...")
        start_time = asyncio.get_event_loop().time()
        
        # Click Bulk CSV tab
        tabs = page.locator("button[role='tab']")
        await move_and_click(page, tabs.nth(1), pause_before=0.4)
        await page.wait_for_timeout(4000)
        
        # Scroll down to show the file upload area
        await scroll_smoothly(page, 200)
        await page.wait_for_timeout(2000)
        
        # Upload the CSV file
        print("Uploading bulk CSV file...")
        file_input = page.locator("input[type='file']")
        await file_input.set_input_files("scratch/test_bulk.csv")
        await page.wait_for_timeout(2000) # Pause 2s after upload
        
        # Click the "Trigger Fleet Campaign Generation" button
        print("Clicking Trigger Fleet Campaign Generation button...")
        trigger_btn = page.get_by_role("button", name="Trigger Fleet Campaign Generation")
        await move_and_click(page, trigger_btn, pause_before=0.6)
        await page.wait_for_timeout(6000) # Wait for batch response and grid to render
        
        # Scroll down to show job tracker table appearing
        await scroll_smoothly(page, 600)
        await page.wait_for_timeout(3000)
        
        # As each row updates, scroll the table to make sure all columns are visible
        # We can drag/scroll Streamlit dataframes or just move mouse to show context
        await scroll_smoothly(page, 800)
        await page.wait_for_timeout(4000)
        
        # Hover over the data editor grid container to highlight the failed row
        data_grid = page.locator("[data-testid='stDataFrame']").first
        await data_grid.scroll_into_view_if_needed()
        box = await data_grid.bounding_box()
        if box:
            print("Hovering over failed row in bulk grid...")
            # Move mouse to the second data row (failed row) of the data editor
            # Header is ~40px, Row 1 is ~35px, Row 2 starts at ~75px. So y = box['y'] + 90
            await page.mouse.move(box['x'] + box['width']/2, box['y'] + 90, steps=25)
            await asyncio.sleep(2.0)
        await page.wait_for_timeout(1000)
        
        # Scroll down to show the two successful rows completing
        await scroll_smoothly(page, 1100)
        await page.wait_for_timeout(4000)
        
        # Hover over the download link of a completed row
        if box:
            print("Hovering over download cell in first row...")
            # Row 1 is B0CHX1W1XY (completed). Move mouse to Row 1, and over to the rightmost columns (Download link is at the right edge)
            await page.mouse.move(box['x'] + box['width'] - 120, box['y'] + 55, steps=25)
            await asyncio.sleep(1.5)
        await page.wait_for_timeout(2000)
        
        # Scroll up and down to show bulk dashboard context
        await scroll_smoothly(page, 400)
        await page.wait_for_timeout(2000)
        await scroll_smoothly(page, 1100)
        await page.wait_for_timeout(2000)
        
        # Maintain exact timing of 108 seconds for Section 5
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 108.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # ==========================================
        # SECTION 6: Architecture in Browser (7:39 to 9:00 - 81s)
        # ==========================================
        print("Starting Section 6: Architecture in Browser...")
        start_time = asyncio.get_event_loop().time()
        
        # Open README in new browser tab
        await page.goto("http://127.0.0.1:8080/README.md")
        await page.wait_for_timeout(5000)
        
        # Scroll down slowly through architecture section
        await scroll_smoothly(page, 500)
        await page.wait_for_timeout(2000)
        
        # Pause 2 seconds on the system flowchart
        await scroll_smoothly(page, 1000)
        await page.wait_for_timeout(2000) # Flowchart pause
        
        # Scroll to Known Limitations section, pause 2 seconds
        await scroll_smoothly(page, 1900)
        await page.wait_for_timeout(2000) # Limitations pause
        
        # Scroll back up to show the tech stack table
        await scroll_smoothly(page, 300)
        await page.wait_for_timeout(3000)
        
        # Slow scroll all the way to the bottom
        for i in range(4, 25):
            await page.evaluate(f"window.scrollTo(0, {i * 120})")
            await page.wait_for_timeout(2500)
            
        # Maintain exact timing of 81 seconds for Section 6
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = 81.0 - elapsed
        if remaining > 0:
            await page.wait_for_timeout(remaining * 1000)
            
        # Close Browser and save video
        print("Closing browser...")
        video_path = await page.video.path()
        await browser.close()
        
        print(f"Playwright video recorded successfully at: {video_path}")
        
        # Move raw video to a known location
        shutil.move(video_path, "scratch/raw_recording.mp4")
        print("Raw recording saved as scratch/raw_recording.mp4")
        
    # Terminate HTTP server
    print("Terminating HTTP server...")
    http_server.terminate()
    http_server.wait()
    print("HTTP server stopped.")

if __name__ == "__main__":
    asyncio.run(record())
